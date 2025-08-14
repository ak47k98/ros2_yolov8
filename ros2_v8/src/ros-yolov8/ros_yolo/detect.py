"""
// 将当前状态发布到currentstate 1=circle:shot/sco 2=h:land
inline int fly_state_to_int(FlyState state) {
  switch (state) {
    case FlyState::init: return 1;
    case FlyState::takeoff: return 1;
    case FlyState::end: return 2;
    case FlyState::Goto_shotpoint: return 1;
    case FlyState::Doshot: return 0;
    case FlyState::Goto_scoutpoint: return 1;
    case FlyState::Surround_see: return 3;
    case FlyState::Doland: return 4;
    case FlyState::Print_Info: return 1;
    default: return 1;
  }
}
#飞行状态
"""

from std_msgs.msg import Int32
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image, Range
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
import threading
import time
import numpy as np
from ros_yolo.servo_controller import ServoController
from vision_msgs.msg import Detection2DArray, Detection2D, BoundingBox2D, ObjectHypothesisWithPose, Pose2D
from geometry_msgs.msg import Point
from std_msgs.msg import Header
from visualization_msgs.msg import MarkerArray, Marker
import queue  # 导入队列模块
import torch

try:
    torch.multiprocessing.set_start_method('fork', force=True)
    print("PyTorch multiprocessing start method has been set to 'fork'.")
except RuntimeError:
    # 如果启动方法已经被设置，会抛出RuntimeError，可以安全地忽略
    print("PyTorch multiprocessing start method was already set.")


# ==================== 新增：健壮的视频流处理器 ====================
class RobustStreamer:


    def __init__(self, stream_url, logger, max_queue_size=1):
        self.stream_url = stream_url
        self.logger = logger
        # 使用固定大小为1的队列，确保只处理最新帧，防止内存堆积
        self.frames_queue = queue.Queue(maxsize=max_queue_size)
        self.running = False
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)

    def _capture_loop(self):
        while self.running:
            self.logger.info(f"[Streamer] 正在尝试连接到: {self.stream_url}...")
            cap = cv2.VideoCapture(self.stream_url)
            if not cap.isOpened():
                self.logger.warn("[Streamer] 连接失败，5秒后重试...")
                time.sleep(5)
                continue

            self.logger.info("[Streamer] 连接成功，开始拉流。")
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    self.logger.warn("[Streamer] 视频流断开或读取失败，准备重连...")
                    break  # 跳出内层循环，触发重连

                # 队列已满，说明处理速度跟不上，丢弃旧帧
                if self.frames_queue.full():
                    try:
                        self.frames_queue.get_nowait()
                    except queue.Empty:
                        pass
                # 将最新帧放入队列
                self.frames_queue.put(frame)

            cap.release()
            time.sleep(1)  # 重连前稍作等待

    def start(self):
        self.running = True
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()
        self.logger.info("[Streamer] 已停止。")

    def get_frame(self):
        """非阻塞地从队列获取最新帧"""
        try:
            return self.frames_queue.get_nowait()
        except queue.Empty:
            return None


# =================================================================

class AIDetector(Node):
    """智能检测节点（摄像头/RTSP/RTMP/ROS话题自动切换，统一处理流程）"""

    def __init__(self):
        super().__init__('detector')

        # ========== 参数声明 ==========
        self.declare_parameters(
            namespace='',
            parameters=[
                ('camera_id', ''),
                ('image_topic', 'image_topic'),
                ('model_path1', 'best_circle.pt'),
                ('model_path2', 'best_H.pt'),
                ('conf_threshold', 0.6),
                ('device', 'cuda:0'),
                ('frame_size', [1920, 1080]),
                ('publish_raw', True),
                ('processing_fps', 30.0),  # 新增：目标处理帧率
            ]
        )

        self.bridge = CvBridge()
        self.camera_id = self.get_parameter('camera_id').value
        self.image_topic = self.get_parameter('image_topic').value

        # ========== 输入源初始化 (重构) ==========
        self.streamer = None
        self.image_sub = None
        self.processing_timer = None

        # ========== 模型初始化 ==========
        self._init_model()

        # ... (您其余的初始化代码保持不变) ...
        self._init_class_mapping()
        self._init_publishers()
        self.visualization_targets = []
        self.visualization_subscriber = self.create_subscription(
            MarkerArray, 'visualization_targets', self._visualization_callback, 10
        )
        self.center_1x, self.center_1y = 700, 450
        self.center_2x, self.center_2y = 610, 450
        self.radius = 35
        self.prev_state = 0
        self.stay_start_time = None
        self.stay_duration_threshold = 1.5
        self.last_servo_value = 0
        self.sum_servo_value = 0
        self.pause_until = None
        self._load_camera_calibration('rgb_camera_calib_1.npz')
        h, w = 720, 1280
        if self.camera_matrix is not None and self.dist_coeffs is not None:
            self.map1, self.map2 = cv2.initUndistortRectifyMap(
                self.camera_matrix, self.dist_coeffs, None,
                self.camera_matrix, (w, h), cv2.CV_16SC2
            )
        else:
            self.map1, self.map2 = None, None
        self.current_state = 0
        self.state_sub = self.create_subscription(
            Int32, 'current_state', self._state_callback, 10
        )
        self.last_h_detected_in_doland = None
        self.last_h_detected_time = None
        self.h_detection_active = False
        self.servo_ctrl = None
        self.servo_ready = False
        self.rangefinder_height = None
        self.rangefinder_sub = self.create_subscription(
            Range, '/mavros/rangefinder/rangefinder', self._range_callback, 10
        )
        threading.Thread(target=self._try_init_servo_controller, daemon=True).start()
        # =================================================

        # ========== 启动输入源处理 ==========
        self._start_input_source()
        self.current3_time=time.time()

    def _start_input_source(self):
        """根据配置选择并启动视频源"""
        processing_fps = self.get_parameter('processing_fps').value
        timer_period = 1.0 / processing_fps

        if self.camera_id:
            self.get_logger().info(f"视频流模式已启动，目标处理帧率: {processing_fps} FPS.")
            self.streamer = RobustStreamer(self.camera_id, self.get_logger())
            self.streamer.start()
            self.processing_timer = self.create_timer(timer_period, self._process_stream_frame)
        else:
            self.get_logger().info(f"ROS话题模式已启动: {self.image_topic}")
            self._init_ros_image_subscriber()

    def _init_ros_image_subscriber(self):
        qos = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT)
        self.image_sub = self.create_subscription(
            Image, self.image_topic, self._ros_image_callback, qos
        )

    def _process_stream_frame(self):
        """定时器回调函数，用于处理来自健壮视频流的帧"""
        if not self.streamer:
            return

        frame = self.streamer.get_frame()
        if frame is not None:
            # 发布原始图像
            if self.raw_pub:
                try:
                    msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
                    self.raw_pub.publish(msg)
                except Exception as e:
                    self.get_logger().error(f"原始图像发布失败: {str(e)}")

            # 处理帧
            self._process_frame(frame)

    def _ros_image_callback(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            # 在ROS模式下，也直接调用核心处理函数
            self._process_frame(frame)
        except Exception as e:
            self.get_logger().error(f"ROS图像转换失败: {e}")

    # 移除了旧的 _init_threads 和 _capture_loop 方法
    # _process_frame 及其后续所有方法保持不变


    def _try_init_servo_controller(self):
        try:
            self.servo_ctrl = ServoController(self, namespace="/mavros/")
            self.servo_ready = True
            self.get_logger().info("舵机控制器初始化完成")
        except Exception as e:
            self.get_logger().warn(f"MAVROS 未启动，舵机控制不可用：{e}")
            self.servo_ready = False

    def _init_model(self):
        device = self.get_parameter('device').value
        try:
            model_path1 = self.get_parameter('model_path1').value
            model_path2 = self.get_parameter('model_path2').value
            self.model1 = YOLO(model_path1).to(device)
            self.model2 = YOLO(model_path2).to(device)
            self.model1.fuse()
            self.model2.fuse()
            self.get_logger().info("Converting fused models to FP16...")
            self.model1.half()
            self.model2.half()
            self.get_logger().info(f"已加载模型1: {model_path1} → {device}")
            self.get_logger().info(f"已加载模型2: {model_path2} → {device}")
        except Exception as e:
            self.get_logger().error(f"模型加载失败: {str(e)}")
            raise

    def _load_camera_calibration(self, path):
        try:
            calib_data = np.load(path)
            self.camera_matrix = calib_data['camera_matrix']
            self.dist_coeffs = calib_data['dist_coeffs']
            self.get_logger().info("成功加载相机标定参数")
            self.get_logger().info(f"相机内参 fx: {self.camera_matrix[0, 0]}, fy: {self.camera_matrix[1, 1]}")
            self.get_logger().info(f"相机畸变系数: {self.dist_coeffs}")
        except Exception as e:
            self.get_logger().error(f"加载标定参数失败: {str(e)}")
            self.camera_matrix = None
            self.dist_coeffs = None

    def _init_publishers(self):
        qos = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT)
        if self.get_parameter('publish_raw').value:
            self.raw_pub = self.create_publisher(Image, 'raw_images', qos)
        else:
            self.raw_pub = None
        self.det2d_pub = self.create_publisher(Detection2DArray, 'detection2d_array', 10)
        self.servo_pub = self.create_publisher(Int32, 'servo_state', 10)

    def _process_frame(self, frame):
        start_time = time.time()
        # 畸变矫正
        if self.camera_matrix is not None and self.dist_coeffs is not None and self.map1 is not None:
            frame = cv2.remap(frame, self.map1, self.map2, interpolation=cv2.INTER_LINEAR)
        conf_thres = self.get_parameter('conf_threshold').value
        try:
            results1 = self.model1.predict(source=frame, conf=conf_thres, verbose=False, stream=False)
            results2 = self.model2.predict(source=frame, conf=conf_thres, verbose=False, stream=False)
            combined_results = results1 + results2

            detected_coords = []
            for result in combined_results:
                for box in result.boxes.cpu().numpy():
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    original_cls_id = int(box.cls[0])
                    label_name = result.names[original_cls_id]
                    unified_cls_id = self.get_unified_class_id(label_name)
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    w = x2 - x1
                    h = y2 - y1
                    detected_coords.append(
                        f"{label_name}(id:{unified_cls_id},cx:{cx},cy:{cy})"
                    )

            annotated_frame = self._draw_detections(frame, combined_results)
            for target in self.visualization_targets:
                self._draw_visualization_target(annotated_frame, target)

            # 保持窗口显示逻辑
            try:
                window_width, window_height = 1920, 1080
                resized_frame = cv2.resize(annotated_frame, (window_width, window_height))
                cv2.imshow('Detection', resized_frame)
                cv2.waitKey(1)
            except Exception:
                # 在无头服务器上运行时，cv2.imshow会失败，可以安全地忽略
                pass

            duration_ms = (time.time() - start_time) * 1000
            coords_info = " | 检测到: " + ", ".join(detected_coords) if detected_coords else ""
            self.get_logger().info(
                f"处理时长: {duration_ms:.1f}ms | 状态: {self.current_state} | servo: {self.last_servo_value}{coords_info}",
                throttle_duration_sec=0.33
            )
        except Exception as e:
            self.get_logger().error(f"推理过程出错: {str(e)}")

    def _draw_detections(self, frame, results):
        original_frame = frame.copy()
        cv2.circle(frame, (self.center_1x, self.center_1y), self.radius, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(frame, (self.center_2x, self.center_2y), self.radius, (0, 255, 255), 2, cv2.LINE_AA)
        det_arr = Detection2DArray()
        det_arr.header.stamp = self.get_clock().now().to_msg()
        det_arr.header.frame_id = 'camera_frame'

        circle_boxes = []
        stuffed_boxes = []
        h_boxes = []

        for result in results:
            for box in result.boxes.cpu().numpy():
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                original_cls_id = int(box.cls[0])
                label_name = result.names[original_cls_id]
                unified_cls_id = self.get_unified_class_id(label_name)
                conf = float(box.conf[0])
                area = (x2 - x1) * (y2 - y1)

                if label_name == 'circle':
                    circle_boxes.append((area, x1, y1, x2, y2, unified_cls_id, conf))
                elif label_name == 'stuffed':
                    stuffed_boxes.append((area, x1, y1, x2, y2, unified_cls_id, conf))
                elif label_name == 'H':
                    h_boxes.append((area, x1, y1, x2, y2, unified_cls_id, conf))

                color = (0, 255, 0)
                if label_name == 'circle':
                    color = (255, 0, 0)
                    try:
                        if self.rangefinder_height is not None:
                            # 标定分辨率
                            calib_w, calib_h = 640, 640
                            # 实际分辨率
                            img_w, img_h = 1280, 720

                            # 计算缩放比例
                            scale_x = img_w / calib_w
                            scale_y = img_h / calib_h

                            # 修正相机内参（只需一次，建议在加载标定参数后做）
                            fx = self.camera_matrix[0, 0] * scale_x
                            cx0 = self.camera_matrix[0, 2] * scale_x
                            cy0 = self.camera_matrix[1, 2] * scale_y

                            # 目标像素中心与宽高
                            cx = (x1 + x2) // 2
                            cy = (y1 + y2) // 2
                            w = x2 - x1
                            h = y2 - y1
                            pixel_diameter = (w + h) / 2
                            # 高度修正
                            height_m = self.rangefinder_height-0.1
                            # 调用估算函数
                            real_diameter = self.estimate_real_diameter_with_angle_correction(
                                pixel_diameter, cx, cy, fx, cx0, cy0, height_m
                            )

                            # 分类目标 circle 类型
                            
                            if 0.125 <= real_diameter <= 0.175:
                                type_str = "15cm"
                            elif 0.175 <= real_diameter <= 0.225:
                                type_str = "20cm"
                            elif 0.225 <= real_diameter <= 0.325:
                                type_str = "25cm"
                            else:
                                type_str = "未知"


                            self.get_logger().info(
                                f"[直径估算] ({cx},{cy}) d={real_diameter * 100:.1f}cm → {type_str}"
                                f" | 高度: ({height_m+0.1})",
                                throttle_duration_sec=1.0
                            )

                    except Exception as e:
                        self.get_logger().warn(f"[直径估算] 跳过估算: {e}", throttle_duration_sec=1.0)



                elif label_name == 'stuffed':
                    color = (0, 255, 255)
                elif label_name == 'H':
                    color = (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label_name} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)



        if self.current_state == 3:
            idx = 0
            for result in results:  # 遍历检测结果
                for box in result.boxes.cpu().numpy():  # 遍历每个检测框
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    original_cls_id = int(box.cls[0])
                    label_name = result.names[original_cls_id]
                    unified_cls_id = self.get_unified_class_id(label_name)
                    conf = float(box.conf[0])
                    if label_name in ['circle', 'H']:
                        cx = (x1 + x2) // 2
                        cy = (y1 + y2) // 2
                        w = x2 - x1
                        h = y2 - y1
                        det2d = self.build_detection2d(cx, cy, w, h, unified_cls_id, conf, det_arr.header.stamp,
                                                    det_arr.header.frame_id)
                        det_arr.detections.append(det2d)


                    if label_name != 'H':
                        roi = original_frame[y1:y2, x1:x2]
                        if roi.size == 0:
                            continue
                        elif roi.size > 0:
                            win = f'Stuffed_{idx}'
                            resized_roi = cv2.resize(roi, (160, 160))   # 调整裁剪区域大小
                            cv2.imshow(win, resized_roi)    # 显示裁剪区域
                            cv2.moveWindow(win, 150 + idx * 180, 50) # 设置窗口位置
                            idx += 1    # 增加索引以避免窗口名称冲突


            self.det2d_pub.publish(det_arr)
            return frame



        elif self.current_state == 4:
            nowtime = float(time.time())
            if self.current3_time >= nowtime - 3.0 and self.current3_time is not None :
                idx = 0
                for result in results:
                    for box in result.boxes.cpu().numpy():  # 遍历每个检测框
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        original_cls_id = int(box.cls[0])
                        label_name = result.names[original_cls_id]
                        unified_cls_id = self.get_unified_class_id(label_name)
                        conf = float(box.conf[0])
                        if label_name == 'stuffed':
                            roi = original_frame[y1:y2, x1:x2]
                            if roi.size == 0:
                                continue
                            elif roi.size > 0:
                                win = f'Stuffed_{idx}'
                                resized_roi = cv2.resize(roi, (160, 160))  # 调整裁剪区域大小
                                cv2.imshow(win, resized_roi)  # 显示裁剪区域
                                cv2.moveWindow(win, 150 + idx * 180, 50)  # 设置窗口位置
                                idx += 1  # 增加索引以避免窗口名称冲突


            det_list = []
            if h_boxes:
                _, x1, y1, x2, y2, cls_id, conf = max(h_boxes, key=lambda b: b[0])
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                w = x2 - x1
                h = y2 - y1
                det_list.append((cls_id, cx, cy, w, h, conf))
                self.last_h_detected_in_doland = (cls_id, cx, cy, w, h, conf)
                self.last_h_detected_time = time.time()
            elif self.last_h_detected_in_doland:
                det_list.append(self.last_h_detected_in_doland)
            for cls_id, cx, cy, w, h, conf in det_list:
                det2d = self.build_detection2d(cx, cy, w, h, cls_id, conf, det_arr.header.stamp,
                                            det_arr.header.frame_id)
                det_arr.detections.append(det2d)
            self.det2d_pub.publish(det_arr)
            return frame

        else:
            for area, x1, y1, x2, y2, cls_id, conf in circle_boxes:
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                w = x2 - x1
                h = y2 - y1
                det2d = self.build_detection2d(cx, cy, w, h, cls_id, conf, det_arr.header.stamp,
                                            det_arr.header.frame_id)
                det_arr.detections.append(det2d)
            for area, x1, y1, x2, y2, cls_id, conf in stuffed_boxes:
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                w = x2 - x1
                h = y2 - y1
                det2d = self.build_detection2d(cx, cy, w, h, cls_id, conf, det_arr.header.stamp,
                                            det_arr.header.frame_id)
                det_arr.detections.append(det2d)
            if circle_boxes and (self.pause_until is None or time.time() >= self.pause_until):
                target_center_x = (self.center_1x + self.center_2x) // 2
                target_center_y = (self.center_1y + self.center_2y) // 2
                min_distance = float('inf')
                closest_circle = None
                for area, x1, y1, x2, y2, cls_id, conf in circle_boxes:
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    distance = ((cx - target_center_x) ** 2 + (cy - target_center_y) ** 2) ** 0.5
                    if distance < min_distance:
                        min_distance = distance
                        closest_circle = (area, x1, y1, x2, y2, cls_id, conf)
                if closest_circle:
                    _, x1, y1, x2, y2, cls_id, conf = closest_circle
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    w = x2 - x1
                    h = y2 - y1
                    d1x = cx - self.center_1x
                    d1y = cy - self.center_1y
                    d2x = cx - self.center_2x
                    d2y = cy - self.center_2y
                    in_target_area = (
                        d1x * d1x + d1y * d1y <= self.radius * self.radius or
                        d2x * d2x + d2y * d2y <= self.radius * self.radius
                    )
                    altitude_ok = (
                        self.rangefinder_height is None or
                        self.rangefinder_height <= 1.6
                    )
                    current_time = time.time()
                    if in_target_area and self.current_state == 0 and altitude_ok and self.last_servo_value != 3:
                        if self.stay_start_time is None:
                            self.stay_start_time = current_time
                            self.get_logger().info("开始计时：目标在圆内，且高度满足")
                        else:
                            elapsed = current_time - self.stay_start_time
                            self.get_logger().info(f"计时中：{elapsed:.1f}s / {self.stay_duration_threshold}s",
                                                throttle_duration_sec=0.33)
                    else:
                        if self.stay_start_time is not None:
                            self.get_logger().info("计时中断，条件不满足，重置计时器")
                        self.stay_start_time = None
                    if self.stay_start_time and (
                        time.time() - self.stay_start_time >= self.stay_duration_threshold
                    ) and self.current_state == 0:
                        if self.rangefinder_height is not None and self.rangefinder_height >= 1.6:
                            self.get_logger().warn(
                                f"[LIDAR] 当前高度为 {self.rangefinder_height:.2f} m，超过投弹限制（<1.6m），跳过投弹"
                            )
                            self.stay_start_time = None
                        elif self.rangefinder_height is None:
                            self.get_logger().warn(
                                f"[LIDAR] 当前激光雷达无读数，跳过投弹"
                            )
                        else:
                            if d1x * d1x + d1y * d1y <= self.radius * self.radius and self.last_servo_value != 1 and self.sum_servo_value != 2:
                                self.get_logger().info("右舵投弹！！！")
                                servo_id = 11
                                self.last_servo_value = 1
                                self.sum_servo_value += 1
                                self.stay_start_time = None
                                try:
                                    if self.servo_ready and hasattr(self, 'servo_ctrl'):
                                        self.servo_ctrl.fire_servo(servo_id)
                                    else:
                                        self.get_logger().warn(f"[Servo] MAVROS未启动，跳过 Servo{servo_id} 投弹动作")
                                except Exception as e:
                                    self.get_logger().error(f"[Servo] 投弹执行出错: {e}")
                                if self.sum_servo_value == 2:
                                    self.get_logger().info("已完成左右舵投弹")
                                    self.last_servo_value = 3
                            elif self.last_servo_value != 2 and self.sum_servo_value != 2 and d2x * d2x + d2y * d2y <= self.radius * self.radius:
                                self.get_logger().info("左舵投弹！！！")
                                servo_id = 12
                                self.last_servo_value = 2
                                self.sum_servo_value += 1
                                self.stay_start_time = None
                                try:
                                    if self.servo_ready and hasattr(self, 'servo_ctrl'):
                                        self.servo_ctrl.fire_servo(servo_id)
                                    else:
                                        self.get_logger().warn(f"[Servo] MAVROS未启动，跳过 Servo{servo_id} 投弹动作")
                                except Exception as e:
                                    self.get_logger().error(f"[Servo] 投弹执行出错: {e}")

                                if self.sum_servo_value == 2:
                                    self.get_logger().info("已完成左右舵投弹")
                                    self.last_servo_value = 3
                            elif self.sum_servo_value == 2:
                                self.last_servo_value = 3
                                self.get_logger().info("已完成前后舵投弹，servo=3 发送完成")
                            self.stay_start_time = None
            if h_boxes:
                _, x1, y1, x2, y2, cls_id, conf = max(h_boxes, key=lambda b: b[0])
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                w = x2 - x1
                h = y2 - y1
                det2d = self.build_detection2d(cx, cy, w, h, cls_id, conf, det_arr.header.stamp,
                                            det_arr.header.frame_id)
                det_arr.detections.append(det2d)
            self.det2d_pub.publish(det_arr)
            self.servo_pub.publish(Int32(data=self.last_servo_value))
            return frame

        #self.det2d_pub.publish(det_arr)
        #self.servo_pub.publish(Int32(data=self.last_servo_value))

        #return frame

    def destroy_node(self):
        """重构后的销毁函数"""
        self.get_logger().info("开始销毁节点...")
        # 停止视频流处理器
        if self.streamer:
            self.streamer.stop()

        # 停止定时器
        if self.processing_timer and not self.processing_timer.cancelled:
            self.processing_timer.cancel()

        # 销毁所有OpenCV窗口
        cv2.destroyAllWindows()

        # 调用父类的销毁方法
        super().destroy_node()
        self.get_logger().info("节点已成功销毁。")


    def _visualization_callback(self, msg: MarkerArray):
        new_targets = []    # 创建一个新的目标列表
        for marker in msg.markers:
            # 仅处理类型为 CYLINDER 且操作为 ADD 的标记
            if marker.type == Marker.CYLINDER and marker.action == Marker.ADD:
                try:
                    # 提取标记信息并存储到目标列表
                    target_info = {
                        'id': int(marker.id), 'category': str(marker.ns),
                        'x': float(marker.pose.position.x), 'y': float(marker.pose.position.y),
                        'radius': max(1.0, float(marker.scale.x / 2.0)),
                        'color': (int(marker.color.b * 255), int(marker.color.g * 255), int(marker.color.r * 255))
                    }
                    new_targets.append(target_info)
                except Exception as e:
                    self.get_logger().warn(f'解析可视化标记时出错: {e}')
        self.visualization_targets = new_targets
        if not new_targets:
            # 如果目标列表为空，记录信息日志
            self.get_logger().info("收到空的可视化目标，清除预测。",
                                   throttle_duration_sec=5.0)

    def _draw_visualization_target(self, image, target):
        try:
            center_x = int(target['x'])
            center_y = int(target['y'])
            radius = int(target['radius'])
            color = target['color']
            cv2.circle(image, (center_x, center_y), radius, color, 2)
            cv2.circle(image, (center_x, center_y), 3, color, -1)
            label = f"Pred_ID:{target['id']} ({target['category']})"
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            text_origin = (center_x - text_width // 2, center_y - radius - 10)
            cv2.rectangle(image,
                          (text_origin[0] - 2, text_origin[1] - text_height - 5),
                          (text_origin[0] + text_width + 2, text_origin[1] + 5),
                          (0, 0, 0), cv2.FILLED)
            cv2.putText(image, label, (text_origin[0], text_origin[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        except Exception as e:
            self.get_logger().error(f"绘制可视化目标ID {target.get('id', 'N/A')} 时失败: {e}")

    def _range_callback(self, msg: Range):
        try:
            self.rangefinder_height = msg.range
        except Exception as e:
            pass

    def build_detection2d(self, cx, cy, w, h, cls_id, conf, header_stamp, header_frame_id):
        # 创建一个 Detection2D 对象
        det2d = Detection2D()
        det2d.header.stamp = header_stamp   # 设置消息头的时间戳
        det2d.header.frame_id = header_frame_id # 设置消息头的帧 ID
        det2d.bbox = BoundingBox2D()        # 创建一个二维边界框对象
        # 设置边界框的中心点
        pose = Pose2D()
        pose.position.x = float(cx)         # 中心点的 x 坐标
        pose.position.y = float(cy)
        pose.theta = 0.0                    # 中心点的角度（默认为 0）
        det2d.bbox.center = pose
        det2d.bbox.size_x = float(w)        # 边界框的宽度
        det2d.bbox.size_y = float(h)        # 边界框的高度
        # 创建一个 ObjectHypothesisWithPose 对象
        hypo = ObjectHypothesisWithPose()
        if cls_id == 0:
            hypo.hypothesis.class_id = str('circle')
        elif cls_id == 2:
            hypo.hypothesis.class_id = str('h')
        elif cls_id == 1:
            hypo.hypothesis.class_id = str('stuffed')
        else:
            hypo.hypothesis.class_id = str(cls_id)
        hypo.hypothesis.score = conf     # 设置置信度分数
        det2d.results = [hypo]          # 将假设结果添加到 Detection2D 对象中
        return det2d                    # 返回 Detection2D 对象

    def _state_callback(self, msg: Int32):
        # 更新当前状态
        self.current_state = msg.data
        if self.current_state != self.prev_state:
            # 如果状态发生变化，记录日志
            self.get_logger().info(f"接收到状态更新: {self.current_state}", throttle_duration_sec=1.0)
            if self.prev_state==3 and self.current_state==4:
                self.current3_time = float(time.time())
            self.prev_state = self.current_state

        if self.current_state == 4:
            # 如果在状态doland下，启用H的记忆
            self.h_detection_active = True
        else:
            self.h_detection_active = False
            #self.last_h_detected_in_doland = None
    def _init_class_mapping(self):
        self.unified_class_mapping = {
            'circle': 0,
            'stuffed': 1,
            'H': 2,
        }
        # 反向映射：从类别ID到标签名称
        self.id_to_label = {v: k for k, v in self.unified_class_mapping.items()}
        self.get_logger().info(f"统一类别映射: {self.unified_class_mapping}")

    def get_unified_class_id(self, label_name: str) -> int:
        return self.unified_class_mapping.get(label_name, -1)

    def estimate_real_diameter_with_angle_correction(self, pixel_diameter, cx, cy, fx, cx0, cy0, height_m):
        import math
        r = math.sqrt((cx - cx0) ** 2 + (cy - cy0) ** 2)
        theta = math.atan(r / fx)
        cos_theta = math.cos(theta)
        if cos_theta == 0:
            return 0
        return (pixel_diameter * height_m) / (fx * cos_theta)


def main(args=None):
    rclpy.init(args=args)
    node = AIDetector()
    try:

        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()