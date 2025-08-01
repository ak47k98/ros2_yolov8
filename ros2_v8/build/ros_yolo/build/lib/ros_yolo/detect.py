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

"""
std_msgs.msg.Int32
用于发布和订阅飞控状态（current_state）、舵机状态（servo_state）。
sensor_msgs.msg.Image
用于订阅和发布图像数据（如摄像头图像、原始图像）。
sensor_msgs.msg.Range
用于订阅激光雷达高度数据（/mavros/rangefinder/rangefinder）。
vision_msgs.msg.Detection2DArray、Detection2D、BoundingBox2D、ObjectHypothesisWithPose、Pose2D
用于目标检测结果的发布（如检测到的目标框、类别、置信度等）。
geometry_msgs.msg.Point
用于处理点坐标（虽然在代码中未直接使用，但已导入）。
std_msgs.msg.Header
用于消息头（如时间戳、帧ID）。
visualization_msgs.msg.MarkerArray、Marker
用于可视化目标的订阅（如圆心、半径、颜色等）。
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

class AIDetector(Node):
    """智能检测节点（摄像头/RTSP/RTMP/ROS话题自动切换，统一处理流程）"""

    def __init__(self):
        super().__init__('detector')

        # ========== 参数声明 ==========
        self.declare_parameters(
            namespace='',
            parameters=[
                ('camera_id', ''),
                #('image_topic', 'raw_images'),
                ('image_topic', 'image_topic'),  # 图像订阅话题
                ('model_path1', 'best_circle.pt'),
                ('model_path2', 'best_H.pt'),
                ('conf_threshold', 0.6),
                ('device', 'cuda:0'),
                ('frame_size', [1920, 1080]),
                ('publish_raw', True),
            ]
        )

        self.bridge = CvBridge()
        self.camera_id = self.get_parameter('camera_id').value
        self.image_topic = self.get_parameter('image_topic').value

        # ========== 输入源初始化 ==========
        self.cap = None
        self.image_sub = None
        self.running = False
        self.cap_thread = None
        self.proc_thread = None
        self.latest_frame = None
        self.frame_lock = threading.Lock()



        # ========== 模型初始化 ==========
        self._init_model()

        # ========== 类别映射初始化 ==========
        self._init_class_mapping()

        # ========== ROS接口 ==========
        self._init_publishers()

        # 可视化订阅
        self.visualization_targets = []
        self.visualization_subscriber = self.create_subscription(
            MarkerArray, 'visualization_targets', self._visualization_callback, 10
        )
        self.get_logger().info("Subscribing to '/visualization_targets' for prediction circles.")

        # 圆心相关初始化
        self.center_1x, self.center_1y = 700, 450 # 1280,720
        self.center_2x, self.center_2y = 610, 450
        self.radius = 35
        self.prev_state = 0
        self.stay_start_time = None
        self.stay_duration_threshold = 1.5
        self.last_servo_value = 0
        self.sum_servo_value = 0
        self.pause_until = None

        # 相机标定
        self._load_camera_calibration('rgb_camera_calib_1.npz')
        h, w = 720, 1280
        if self.camera_matrix is not None and self.dist_coeffs is not None:
            self.map1, self.map2 = cv2.initUndistortRectifyMap(
                self.camera_matrix, self.dist_coeffs, None,
                self.camera_matrix, (w, h), cv2.CV_16SC2
            )
        else:
            self.map1, self.map2 = None, None

        # 飞控状态
        self.current_state = 0
        self.state_sub = self.create_subscription(
            Int32, 'current_state', self._state_callback, 10
        )



        # 降落相关变量
        self.last_h_detected_in_doland = None
        self.last_h_detected_time = None
        self.h_detection_active = False


        if self.camera_id:
            self.cap = cv2.VideoCapture(self.camera_id)
            if self.cap.isOpened():
                frame_size = self.get_parameter('frame_size').value
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_size[0])
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_size[1])
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.running = True
                self._init_threads()
                self.get_logger().info(f"视频源模式: {self.camera_id}")
            else:
                self.cap = None
                self.get_logger().warn(f"无法打开视频源 '{self.camera_id}'，自动切换到ROS话题模式")
                self._init_ros_image_subscriber()
        else:
            self._init_ros_image_subscriber()
            self.get_logger().info(f"ROS话题模式: {self.image_topic}")
        # 舵机控制


        #try:
            #self.servo_ctrl = ServoController(self, namespace="/mavros/")
            #self.servo_ready = True
        #except Exception as e:
            #self.get_logger().warn(f"MAVROS 未启动，舵机控制不可用：{e}")
            #self.servo_ready = False

        self.servo_ctrl = None
        self.servo_ready = False
        # 激光雷达订阅
        self.rangefinder_height = None
        self.rangefinder_sub = self.create_subscription(
            Range, '/mavros/rangefinder/rangefinder', self._range_callback, 10
        )
        threading.Thread(target=self._try_init_servo_controller, daemon=True).start()

    def _init_ros_image_subscriber(self):
        qos = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT)
        self.image_sub = self.create_subscription(
            Image, self.image_topic, self._ros_image_callback, qos
        )

    def _try_init_servo_controller(self):
        try:
            self.servo_ctrl = ServoController(self, namespace="/mavros/")
            self.servo_ready = True
            self.get_logger().info("舵机控制器初始化完成")
        except Exception as e:
            self.get_logger().warn(f"MAVROS 未启动，舵机控制不可用：{e}")
            self.servo_ready = False
    def _ros_image_callback(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self._process_frame(frame)
        except Exception as e:
            self.get_logger().error(f"ROS图像转换失败: {e}")

    def _init_model(self):
        device = self.get_parameter('device').value
        try:
            model_path1 = self.get_parameter('model_path1').value
            model_path2 = self.get_parameter('model_path2').value
            self.model1 = YOLO(model_path1).to(device)
            self.model2 = YOLO(model_path2).to(device)
            self.model1.fuse()
            self.model2.fuse()
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

    def _init_threads(self):
        self.running = True
        self.cap_thread = threading.Thread(target=self._capture_loop)
        self.cap_thread.start()

    def _capture_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                self.get_logger().warn("视频帧获取失败", throttle_duration_sec=1)
                continue
            self._process_frame(frame)
            # 发布原始图像
            if self.raw_pub:
                try:
                    msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
                    self.raw_pub.publish(msg)
                except Exception as e:
                    self.get_logger().error(f"图像发布失败: {str(e)}")

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
                        f"{label_name}(orig_id:{original_cls_id},unified_id:{unified_cls_id},cx:{cx},cy:{cy},w:{w},h:{h})"
                    )

            annotated_frame = self._draw_detections(frame, combined_results)
            for target in self.visualization_targets:
                self._draw_visualization_target(annotated_frame, target)
            window_width, window_height = 1920, 1080
            resized_frame = cv2.resize(annotated_frame, (window_width, window_height))
            cv2.imshow('Detection', resized_frame)
            cv2.waitKey(1)

            duration_ms = (time.time() - start_time) * 1000
            coords_info = " | 检测到: " + ", ".join(detected_coords) if detected_coords else ""
            self.get_logger().info(
                f"处理时长: {duration_ms:.1f}ms | 当前状态: {self.current_state} | servo: {self.last_servo_value}{coords_info}",
                throttle_duration_sec=0.33
            )
        except Exception as e:
            self.get_logger().error(f"推理过程出错: {str(e)}")
            time.sleep(0.05)

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
        #idx = 0

        for result in results:
            for box in result.boxes.cpu().numpy():
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                original_cls_id = int(box.cls[0])
                label_name = result.names[original_cls_id]
                unified_cls_id = self.get_unified_class_id(label_name)
                conf = float(box.conf[0])
                area = (x2 - x1) * (y2 - y1)
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                w = x2 - x1
                h = y2 - y1
                if label_name == 'circle':
                    circle_boxes.append((area, x1, y1, x2, y2, unified_cls_id, conf))
                elif label_name == 'stuffed':
                    stuffed_boxes.append((area, x1, y1, x2, y2, unified_cls_id, conf))
                elif label_name == 'H':
                    h_boxes.append((area, x1, y1, x2, y2, unified_cls_id, conf))
                color = (0, 255, 0)
                if label_name == 'circle':
                    color = (255, 0, 0)


                    # 直径估算逻辑
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
                            elif 0.225 <= real_diameter <= 0.275:
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


                    if label_name == 'stuffed':
                        roi = original_frame[y1:y2, x1:x2]
                        if roi.size == 0:
                            continue
                        elif roi.size > 0:
                            win = f'Stuffed_{idx}'
                            resized_roi = cv2.resize(roi, (160, 160))   # 调整裁剪区域大小
                            cv2.imshow(win, resized_roi)    # 显示裁剪区域
                            cv2.moveWindow(win, 50 + idx * 180, 50) # 设置窗口位置
                            idx += 1    # 增加索引以避免窗口名称冲突


            self.det2d_pub.publish(det_arr)
            return frame



        elif self.current_state == 4:
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

    def _visualization_callback(self, msg: MarkerArray):
        new_targets = []    # 创建一个新的目标列表
        for marker in msg.markers:
            # 仅处理类型为 CYLINDER 且操作为 ADD 的标记
            if marker.type == Marker.CYLINDER and marker.action == Marker.ADD:
                try:
                    # 提取标记信息并存储到目标列表
                    target_info = {
                        'id': int(marker.id),
                        'category': str(marker.ns),
                        'x': float(marker.pose.position.x),
                        'y': float(marker.pose.position.y),
                        'radius': max(1.0, float(marker.scale.x / 2.0)),
                        'color': (int(marker.color.b * 255), int(marker.color.g * 255), int(marker.color.r * 255))
                    }
                    new_targets.append(target_info)
                except Exception as e:
                    # 如果解析标记时出错，记录警告日志
                    self.get_logger().warn(f'解析可视化标记时出错: {e}')
        # 更新可视化目标列表
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
            self.get_logger().info(f"接收到状态更新: {self.current_state}", throttle_duration_sec=2.0)
            self.prev_state = self.current_state

        if self.current_state == 4:
            # 如果在状态doland下，启用H的记忆
            self.h_detection_active = True
            self.last_h_detected_in_doland = None
        else:
            self.h_detection_active = False
            self.last_h_detected_in_doland = None

    def destroy_node(self):
        # 停止运行标志
        self.running = False
        # 等待视频捕获线程结束
        if self.cap_thread and self.cap_thread.is_alive():
            self.cap_thread.join()
        # 释放视频捕获资源
        if self.cap:
            self.cap.release()
        # 销毁所有OpenCV窗口
        cv2.destroyAllWindows()
        # 调用父类的销毁方法
        super().destroy_node()

    def _init_class_mapping(self):
        self.unified_class_mapping = {
            'circle': 0,
            'stuffed': 1,
            'H': 2,
        }
        # 反向映射：从类别ID到标签名称
        self.id_to_label = {v: k for k, v in self.unified_class_mapping.items()}
        self.get_logger().info(f"统一类别映射: {self.unified_class_mapping}")

    def get_unified_class_id(self, label_name: str) -> int: # 根据标签名称获取统一类别ID，如果标签名称不存在，则返回-1
        return self.unified_class_mapping.get(label_name, -1)

    def estimate_real_diameter_with_angle_correction(self, pixel_diameter, cx, cy, fx, cx0, cy0, height_m):
        """
        根据视角修正模型估算真实直径

        参数:
            pixel_diameter: 检测框像素直径（宽高均值）
            cx, cy: 目标中心点坐标（像素）
            fx: 相机焦距（像素）
            cx0, cy0: 图像中心点（像素）
            height_m: 高度（米）

        返回:
            估计出的目标直径（单位：米）
        """
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
