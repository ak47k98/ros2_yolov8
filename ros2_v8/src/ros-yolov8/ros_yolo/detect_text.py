from std_msgs.msg import Int32
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
import threading
import time
import numpy as np
from ros_yolo.servo_controller import ServoController
from sensor_msgs.msg import Range
# 修正导入：只从 vision_msgs 导入 Pose2D，从 geometry_msgs 导入 Point
from vision_msgs.msg import Detection2DArray, Detection2D, BoundingBox2D, ObjectHypothesisWithPose, Pose2D
from geometry_msgs.msg import Point
from std_msgs.msg import Header
from visualization_msgs.msg import MarkerArray, Marker


class AIDetector(Node):
    """智能检测节点（从话题订阅图像）"""

    def __init__(self):
        super().__init__('detector')

        # ========== 参数声明 ==========
        self.declare_parameters(
            namespace='',
            parameters=[
                ('model_path1', '/home/ak47k98/PycharmProjects/ros2_v8/best_circle.pt'),
                ('model_path2', '/home/ak47k98/PycharmProjects/ros2_v8/best_H.pt'),
                ('conf_threshold', 0.6),
                ('device', 'cuda:0'),
                #('image_topic', 'image_topic'),  # 图像订阅话题
                ('image_topic', 'raw_images'),  # 图像订阅话题
                ('model_path1', 'best_old_circle.pt'),  # 模型路径1
                ('model_path2', 'best_H.pt'),# 模型路径2
                ('conf_threshold', 0.5),  # 置信度阈值
                ('device', 'cuda:0'),  # 推理设备
                ('frame_size', [1920, 1080]),  # [宽, 高]
            ]
        )

        # ========== 初始化 ==========
        self.bridge = CvBridge()  # OpenCV/ROS图像转换工具
        self._init_model()
        self._init_class_mapping()

        # ========== ROS接口 ==========
        self._init_publishers()
        self._init_subscribers()  # 激活订阅器
        self._init_threads()

        # MODIFICATION: Initialize a list to store (hold) visualization targets
        self.visualization_targets = []

        # MODIFICATION: Add a new subscriber for visualization markers (prediction circles)
        self.visualization_subscriber = self.create_subscription(
            MarkerArray,
            'visualization_targets',  # This topic name should match the external publisher
            self._visualization_callback,
            10
        )
        self.get_logger().info("Subscribing to '/visualization_targets' for prediction circles.")

        # 圆心相关初始化
        self.center_1x, self.center_1y = 710, 450  # 1280,720
        self.center_2x, self.center_2y = 630, 450
        self.radius = 35

        self.prev_state = 0

        self.stay_start_time = None
        self.stay_duration_threshold = 1.5  # 秒

        self.last_servo_value = 0
        self.sum_servo_value = 0

        self.pause_until = None

        self._load_camera_calibration('rgb_camera_calib_1.npz')

        self.current_state = 0  # 初始化飞控状态
        self.state_sub = self.create_subscription(
            Int32,
            'current_state',
            self._state_callback,
            10
        )

        h, w = 720, 1280
        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            self.camera_matrix, self.dist_coeffs, None,
            self.camera_matrix, (w, h), cv2.CV_16SC2
        )
        try:
            self.servo_ctrl = ServoController(self, namespace="/mavros/")
            self.servo_ready = True
        except Exception as e:
            self.get_logger().warn(f"MAVROS 未启动，舵机控制不可用：{e}")
            self.servo_ready = False
        # 降落相关变量（H识别历史）
        self.last_h_detected_in_doland = None
        self.last_h_detected_time = None
        self.h_detection_active = False

        # ========== 激光雷达订阅 ==========
        self.rangefinder_height = None
        self.rangefinder_sub = self.create_subscription(
            Range,
            '/mavros/rangefinder/rangefinder',
            self._range_callback,
            10
        )

    # 移除 @staticmethod
    def build_detection2d(self, cx, cy, w, h, cls_id, conf, header_stamp, header_frame_id):
        det2d = Detection2D()
        det2d.header.stamp = header_stamp
        det2d.header.frame_id = header_frame_id
        det2d.bbox = BoundingBox2D()

        # 修正：使用 vision_msgs.msg.Pose2D
        pose = Pose2D()
        pose.position.x = float(cx)
        pose.position.y = float(cy)
        # z 会默认为0.0，不需要显式设置
        pose.theta = 0.0
        det2d.bbox.center = pose

        det2d.bbox.size_x = float(w)
        det2d.bbox.size_y = float(h)
        hypo = ObjectHypothesisWithPose()
        if cls_id == 0:
            hypo.hypothesis.class_id = str('circle')
        elif cls_id == 2:
            hypo.hypothesis.class_id = str('h')
        elif cls_id == 1:
            hypo.hypothesis.class_id = str('stuffed')
        else:
            hypo.hypothesis.class_id = str(cls_id)
        hypo.hypothesis.score = conf
        det2d.results = [hypo]
        return det2d

    def _init_subscribers(self):
        """初始化图像订阅器"""
        image_topic = self.get_parameter('image_topic').value
        qos = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT)

        self.image_sub = self.create_subscription(
            Image,
            image_topic,
            self.image_callback,  # Callback to handle image
            qos
        )
        self.get_logger().info(f"Subscribing to image topic: '{image_topic}'")

    def image_callback(self, msg: Image):
        """Callback for handling incoming image messages."""
        try:
            # Convert ROS Image message to OpenCV image
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            with self.frame_lock:
                self.latest_frame = frame
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {str(e)}")

    def _state_callback(self, msg: Int32):
        # 飞控状态订阅回调（控制业务主流程分支）
        self.current_state = msg.data
        if self.current_state != self.prev_state:
            self.get_logger().info(f"接收到状态更新: {self.current_state}", throttle_duration_sec=2.0)
            self.prev_state = self.current_state

        if self.current_state == 4:
            self.h_detection_active = True
            self.last_h_detected_in_doland = None
        else:
            self.h_detection_active = False
            self.last_h_detected_in_doland = None

    def _init_model(self):
        # 加载YOLOv8模型
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
        # 加载摄像头标定参数（畸变校正用）
        try:
            calib_data = np.load(path)
            self.camera_matrix = calib_data['camera_matrix']
            self.dist_coeffs = calib_data['dist_coeffs']
            self.get_logger().info("成功加载相机标定参数")
        except Exception as e:
            self.get_logger().error(f"加载标定参数失败: {str(e)}")
            self.camera_matrix = None
            self.dist_coeffs = None

    def _init_publishers(self):
        # 初始化ROS话题发布器
        self.det2d_pub = self.create_publisher(Detection2DArray, 'detection2d_array', 10)
        self.servo_pub = self.create_publisher(Int32, 'servo_state', 10)

    def _init_threads(self):
        # 启动AI推理线程
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.running = True
        # The capture thread is no longer needed
        self.proc_thread = threading.Thread(target=self._process_loop)
        self.proc_thread.start()

    def _visualization_callback(self, msg: MarkerArray):
        """
        处理并存储预测圆标记的回调函数。
        此函数只更新状态，绘制操作在主处理循环中进行。
        """
        new_targets = []
        for marker in msg.markers:
            if marker.type == Marker.CYLINDER and marker.action == Marker.ADD:
                try:
                    # 解析标记信息，并存储为易于使用的字典格式
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
                    self.get_logger().warn(f'解析可视化标记时出错: {e}')

        # 原子性地更新要绘制的目标列表
        self.visualization_targets = new_targets
        # 如果收到空消息，列表会变空，屏幕上的预测圆也会消失，实现了“没有就不绘制”
        if not new_targets:
            self.get_logger().info("Received empty visualization targets, clearing predictions.",
                                   throttle_duration_sec=5.0)

    def _draw_visualization_target(self, image, target):
        """在给定的图像上绘制单个预测圆。"""
        try:
            # 假设标记的坐标已经是像素坐标
            center_x = int(target['x'])
            center_y = int(target['y'])
            radius = int(target['radius'])
            color = target['color']  # 颜色已是BGR元组格式

            # 绘制预测圆的外框
            cv2.circle(image, (center_x, center_y), radius, color, 2)
            # 绘制中心点
            cv2.circle(image, (center_x, center_y), 3, color, -1)

            # 准备并绘制文本标签
            label = f"Pred_ID:{target['id']} ({target['category']})"
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            text_origin = (center_x - text_width // 2, center_y - radius - 10)

            # 为文本添加背景以提高可见性
            cv2.rectangle(image,
                          (text_origin[0] - 2, text_origin[1] - text_height - 5),
                          (text_origin[0] + text_width + 2, text_origin[1] + 5),
                          (0, 0, 0), cv2.FILLED)
            cv2.putText(image, label, (text_origin[0], text_origin[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
                        1)

        except Exception as e:
            self.get_logger().error(f"绘制可视化目标ID {target.get('id', 'N/A')} 时失败: {e}")

    def _range_callback(self, msg: Range):  # 激光雷达距离回调
        self.rangefinder_height = msg.range

    def _process_loop(self):
        # AI推理处理主循环（多线程，核心业务入口）
        while self.running:
            with self.frame_lock:
                frame = self.latest_frame.copy() if self.latest_frame is not None else None

            if frame is None:
                time.sleep(0.01)  # Wait for the first frame from the callback
                continue

            start_time = time.time()
            if self.camera_matrix is not None and self.dist_coeffs is not None:
                frame = cv2.remap(frame, self.map1, self.map2, interpolation=cv2.INTER_LINEAR)
            conf_thres = self.get_parameter('conf_threshold').value
            try:
                results1 = self.model1.predict(source=frame, conf=conf_thres, verbose=False, stream=False)
                results2 = self.model2.predict(source=frame, conf=conf_thres, verbose=False, stream=False)
                combined_results = results1 + results2

                # 收集所有检测到的坐标信息
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
                            f"{label_name}(orig_id:{original_cls_id},unified_id:{unified_cls_id},cx:{cx},cy:{cy},w:{w},h:{h})")
                # 业务逻辑与可视化处理
                annotated_frame = self._draw_detections(frame, combined_results)
                window_width, window_height = 1920, 1080
                resized_frame = cv2.resize(annotated_frame, (window_width, window_height))
                cv2.imshow('Detection', resized_frame)
                cv2.waitKey(1)
                # 日志打印，包括所有目标坐标
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
        """业务逻辑迁移，检测结果用 Detection2DArray 发布，舵机状态用 Int32"""
        # 在图像上绘制检测结果
        cv2.circle(frame, (self.center_1x, self.center_1y), self.radius, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(frame, (self.center_2x, self.center_2y), self.radius, (0, 255, 255), 2, cv2.LINE_AA)
        # 创建检测消息
        det_arr = Detection2DArray()
        det_arr.header.stamp = self.get_clock().now().to_msg()
        det_arr.header.frame_id = 'camera_frame'

        circle_boxes = []
        stuffed_boxes = []
        h_boxes = []
        idx = 0

        # 收集检测框
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
                # 分类收集（使用统一的cls_id）
                if label_name == 'circle':
                    circle_boxes.append((area, x1, y1, x2, y2, unified_cls_id, conf))
                elif label_name == 'stuffed':
                    stuffed_boxes.append((area, x1, y1, x2, y2, unified_cls_id, conf))
                elif label_name == 'H':
                    h_boxes.append((area, x1, y1, x2, y2, unified_cls_id, conf))
                # 可视化
                color = (0, 255, 0)  # 默认绿色
                if label_name == 'circle':
                    color = (255, 0, 0)  # 蓝色
                elif label_name == 'stuffed':
                    color = (0, 255, 255)  # 黄色
                elif label_name == 'H':
                    color = (0, 0, 255)  # 红色
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label_name} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.circle(frame, (cx, cy), 5, color, -1)
                idx += 1

        # 侦查状态
        if self.current_state == 3:
            idx = 0
            # 需要重新遍历results来获取label_name
            for result in results:
                for box in result.boxes.cpu().numpy():
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    original_cls_id = int(box.cls[0])
                    label_name = result.names[original_cls_id]
                    unified_cls_id = self.get_unified_class_id(label_name)
                    conf = float(box.conf[0])

                    # 侦查状态：发布circle和H的坐标信息用于导航
                    if label_name in ['circle', 'H']:
                        cx = (x1 + x2) // 2
                        cy = (y1 + y2) // 2
                        w = x2 - x1
                        h = y2 - y1
                        det2d = self.build_detection2d(cx, cy, w, h, unified_cls_id, conf, det_arr.header.stamp,
                                                       det_arr.header.frame_id)
                        det_arr.detections.append(det2d)

                    # 侦查模式小窗显示：主要看stuffed内容，circle作为防误识别的辅助
                    if label_name in ['stuffed', 'circle']:
                        roi = frame[y1:y2, x1:x2]
                        if roi.size > 0:
                            win = f'{label_name}_{idx}'  # stuffed是主要关注对象，circle是辅助
                            resized_roi = cv2.resize(roi, (160, 160))
                            cv2.imshow(win, resized_roi)
                            cv2.moveWindow(win, 50 + idx * 180, 50)  # 每个窗口水平间隔180px
                            idx += 1

            self.det2d_pub.publish(det_arr)
            return frame

        # 降落状态（只处理H框，历史坐标缓存）
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

        # 投弹及常规状态（0，1，2）
        else:
            # 发布所有circle目标，让控制端根据w、h判断大小并选择合适的目标
            for area, x1, y1, x2, y2, cls_id, conf in circle_boxes:
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                w = x2 - x1
                h = y2 - y1
                det2d = self.build_detection2d(cx, cy, w, h, cls_id, conf, det_arr.header.stamp,
                                               det_arr.header.frame_id)
                det_arr.detections.append(det2d)

            # 投弹逻辑使用距离两个圆心中点最近的circle目标
            if circle_boxes and (self.pause_until is None or time.time() >= self.pause_until):
                # 计算两个圆心的中点
                target_center_x = (self.center_1x + self.center_2x) // 2
                target_center_y = (self.center_1y + self.center_2y) // 2

                # 找到距离中点最近的circle目标
                min_distance = float('inf')
                closest_circle = None
                for area, x1, y1, x2, y2, cls_id, conf in circle_boxes:
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    # 计算到中点的距离
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
                            self.rangefinder_height < 1.6
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
                            elif self.sum_servo_value == 2:
                                self.last_servo_value = 3
                                self.get_logger().info("已完成前后舵投弹，servo=3 发送完成")
                            self.stay_start_time = None

            # stuffed目标不发布ROS消息，仅用于可视化

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

    def destroy_node(self):  # ROS节点销毁，关闭线程与窗口
        self.running = False
        if hasattr(self, 'proc_thread'):
            self.proc_thread.join()
        cv2.destroyAllWindows()
        super().destroy_node()

    def _init_class_mapping(self):
        """初始化统一的类别映射系统，避免双模型cls_id冲突"""
        # 统一的类别映射：label_name -> unified_cls_id
        self.unified_class_mapping = {
            'circle': 0,  # 圆形目标 (来自best_circle.pt)
            'stuffed': 1,  # 投掷物目标 (来自best_circle.pt)
            'H': 2,  # H型降落标识 (来自best_H.pt)
        }

        # 反向映射：unified_cls_id -> label_name
        self.id_to_label = {v: k for k, v in self.unified_class_mapping.items()}

        self.get_logger().info(f"统一类别映射: {self.unified_class_mapping}")

    def get_unified_class_id(self, label_name: str) -> int:
        """根据标签名获取统一的类别ID"""
        return self.unified_class_mapping.get(label_name, -1)  # -1表示未知类别


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