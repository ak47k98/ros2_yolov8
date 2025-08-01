import rclpy
from rclpy.node import Node
from ros2_yolo_msgs.msg import DetectedBox
import time
import cv2

class CoordinateListener(Node):
    """监听检测框坐标并显示在终端，同时计算FPS"""

    def __init__(self):
        super().__init__('coordinate_listener')

        self.last_time = time.time()  # 上次接收坐标的时间
        self.frame_count = 0  # 计数接收到的坐标帧数
        self.fps = 0  # 当前帧率

        # 创建订阅器，订阅检测框坐标
        self.subscription = self.create_subscription(
            DetectedBox,
            'detected_boxes',
            self.listener_callback,
            10
        )
        self.get_logger().info('开始监听检测框坐标...')

    def listener_callback(self, msg):
        """回调函数，处理接收到的坐标数据并显示在终端"""
        # 每接收到一帧坐标，计数加1
        self.frame_count += 1

        # 计算FPS
        current_time = time.time()
        time_diff = current_time - self.last_time
        if time_diff >= 1.0:  # 每秒更新一次FPS
            self.fps = self.frame_count
            self.frame_count = 0  # 重置帧数计数器
            self.last_time = current_time

        # 显示坐标和FPS
        self.get_logger().info(
            f"检测框坐标: ({msg.x1}, {msg.y1}), ({msg.x2}, {msg.y2}),flag:{msg.servo}, FPS: {self.fps:.1f}"#1280.720左上角0.0
        )


def main(args=None):
    rclpy.init(args=args)
    node = CoordinateListener()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
