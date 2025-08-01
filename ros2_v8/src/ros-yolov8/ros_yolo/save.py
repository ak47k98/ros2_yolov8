import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
from datetime import datetime

class ImageSaver(Node):
    def __init__(self):
        super().__init__('image_saver')

        # 保存目录
        self.save_dir = '/home/ak47k98/PycharmProjects/ros2_v8/saved_images'
        os.makedirs(self.save_dir, exist_ok=True)

        self.bridge = CvBridge()

        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT  # ⚠️ 关键点！必须匹配发布者
        )

        self.subscription = self.create_subscription(
            Image,
            'raw_images',  # 订阅 detect.py 中发布的话题
            self.listener_callback,
            qos
        )

        self.get_logger().info("图像保存节点已启动，等待接收图像...")

    def listener_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
            filename = os.path.join(self.save_dir, f"{timestamp}.png")
            cv2.imwrite(filename, frame)
            self.get_logger().info(f"图像已保存: {filename}")
        except Exception as e:
            self.get_logger().error(f"图像保存失败: {str(e)}")

def main(args=None):
    rclpy.init(args=args)
    node = ImageSaver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

