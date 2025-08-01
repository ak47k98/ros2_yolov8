import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
from datetime import datetime


class ImageRecorder(Node):
    def __init__(self):
        super().__init__('image_recorder')

        # 默认的图像话题和保存路径
        self.image_topic = 'raw_images'
        self.output_dir = '/home/ak47k98/PycharmProjects/ros2_v8/video'  # 默认保存路径
        self.frame_rate = 30.0  # 默认帧率

        # 创建目录（如不存在）
        os.makedirs(self.output_dir, exist_ok=True)

        # 使用时间戳生成唯一文件名
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.output_path = os.path.join(self.output_dir, f"{timestamp}.avi")

        self.bridge = CvBridge()
        self.frames = []

        self.subscription = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            10
        )

        self.get_logger().info(f"订阅图像话题: {self.image_topic}")
        self.get_logger().info(f"视频将保存至: {self.output_path}")
        self.get_logger().info("按 Ctrl+C 退出时将拼接为视频")

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.frames.append(cv_image)
        except Exception as e:
            self.get_logger().error(f"图像转换失败: {e}")

    def destroy_node(self):
        super().destroy_node()
        self.get_logger().info("节点正在关闭，正在保存视频...")

        if not self.frames:
            self.get_logger().warn("未接收到任何图像，未生成视频")
            return

        height, width, _ = self.frames[0].shape
        out = cv2.VideoWriter(
            self.output_path,
            cv2.VideoWriter_fourcc(*'XVID'),
            self.frame_rate,
            (width, height)
        )

        for frame in self.frames:
            out.write(frame)
        out.release()

        self.get_logger().info(f"视频保存完成: {self.output_path}")


def main(args=None):
    rclpy.init(args=args)
    node = ImageRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("检测到 Ctrl+C，准备退出")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
