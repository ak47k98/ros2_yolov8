import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection2DArray, Detection2D


class DetectionSubscriber(Node):
    """
    一个订阅Detection2DArray消息并打印详细原始数据的节点。
    """

    def __init__(self):
        super().__init__('detection_subscriber')

        # 创建一个订阅者，监听 'detection2d_array' 话题
        # 注意：话题名称应与你的发布节点（AIDetector）发布的话题名称一致
        self.subscription = self.create_subscription(
            Detection2DArray,
            'detection2d_array',  # 确保这个话题名与发布者一致
            self.listener_callback,
            10)
        self.get_logger().info("Detection subscriber is running and waiting for messages...")

    def listener_callback(self, msg: Detection2DArray):
        """
        回调函数，在收到消息时被调用，并打印消息的全部内容。
        """
        # 打印整个消息数组的头部信息
        self.get_logger().info(
            f"\n--- New Detection Array Received (stamp: {msg.header.stamp.sec}.{msg.header.stamp.nanosec}, frame: '{msg.header.frame_id}') ---"
        )
        self.get_logger().info(f"Total detections in this array: {len(msg.detections)}")

        if not msg.detections:
            self.get_logger().info("--- End of Array ---")
            return

        # 遍历数组中的每一个检测目标
        for i, det in enumerate(msg.detections):
            self.get_logger().info(f"  [Detection #{i}]")

            # 打印 BoundingBox 的详细原始数据
            bbox = det.bbox
            center_pose = bbox.center
            self.get_logger().info(f"    - BBox Center (Pose2D):")
            self.get_logger().info(
                f"        - position (x, y): ({center_pose.position.x:.2f}, {center_pose.position.y:.2f})")
            self.get_logger().info(f"        - theta: {center_pose.theta:.4f}")
            self.get_logger().info(f"    - BBox Size (size_x, size_y): ({bbox.size_x:.2f}, {bbox.size_y:.2f})")

            # 打印所有结果（hypotheses）的详细原始数据
            if det.results:
                self.get_logger().info(f"    - Results ({len(det.results)} hypotheses):")
                for j, result in enumerate(det.results):
                    hypothesis = result.hypothesis
                    self.get_logger().info(f"      - Hypothesis #{j}:")
                    self.get_logger().info(f"          - class_id: '{hypothesis.class_id}'")
                    self.get_logger().info(f"          - score: {hypothesis.score:.4f}")
                    # 也可以打印完整的 result 对象来查看所有原始数据
                    # self.get_logger().info(f"        Raw result object: {result}")
            else:
                self.get_logger().info("    - No results (hypotheses) for this detection.")

        self.get_logger().info("--- End of Array ---")


def main(args=None):
    rclpy.init(args=args)
    node = DetectionSubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # 节点销毁
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()