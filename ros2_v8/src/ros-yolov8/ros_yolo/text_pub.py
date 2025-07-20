import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithScore
from builtin_interfaces.msg import Header
from geometry_msgs.msg import Pose2D
from vision_msgs.msg import BoundingBox2D

import time


class DetectionPublisher(Node):
    def __init__(self):
        super().__init__('detection_publisher')
        self.publisher_ = self.create_publisher(Detection2DArray, 'detections', 10)
        self.timer = self.create_timer(1.0, self.timer_callback)
        self.get_logger().info("模拟 Detection2DArray 发布器已启动")

    def timer_callback(self):
        # 创建消息头
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'camera_frame'

        # 创建一个 detection
        det = Detection2D()
        det.bbox.center.position.x = 320.0
        det.bbox.center.position.y = 240.0
        det.bbox.size_x = 80.0
        det.bbox.size_y = 100.0

        # 添加一个目标预测结果
        hypothesis = ObjectHypothesisWithScore()
        hypothesis.id = 2  # 比如 2 = H
        hypothesis.score = 0.93
        det.results.append(hypothesis)

        # 组合为 Detection2DArray
        msg = Detection2DArray()
        msg.header = header
        msg.detections.append(det)

        # 发布消息
        self.publisher_.publish(msg)
        self.get_logger().info("已发布模拟检测目标: cls_id=2, conf=0.93, (cx=320, cy=240)")


def main(args=None):
    rclpy.init(args=args)
    node = DetectionPublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
