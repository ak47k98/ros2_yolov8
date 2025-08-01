# Copyright 2016 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class CameraCapturer(Node):

    def __init__(self):
        super().__init__('camera_capturer')

        # 创建发布器
        self.publisher_ = self.create_publisher(Image, 'image_topic', 10)

        # 创建定时器，定期发布图像
        timer_period = 0.02  # 20Hz = 0.05秒一次
        self.timer = self.create_timer(timer_period, self.timer_callback)

        # 初始化OpenCV相机
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.get_logger().error("无法打开摄像头，使用备用图像。")
            self.cap = None  # 标记摄像头不可用

        self.bridge = CvBridge()

    def timer_callback(self):
        if self.cap:
            ret, frame = self.cap.read()
            if not ret:
                self.get_logger().warn("摄像头读取失败，使用备用图像。")
                frame = cv2.imread('DJI_20250402174824_0002_V.JPG')  # 备用图
        else:
            frame = cv2.imread('DJI_20250402174824_0002_V.JPG')

        if frame is None:
            self.get_logger().error("找不到fallback.jpg备用图，无法发布图像。")
            return

        # 转换成ROS2的Image消息
        image_topic = self.bridge.cv2_to_imgmsg(frame, "bgr8")
        self.publisher_.publish(image_topic)

        self.get_logger().info('Publishing image', throttle_duration_sec=2)

    def destroy_node(self):
        if self.cap:
            self.cap.release()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)

    camera_capturer = CameraCapturer()

    try:
        rclpy.spin(camera_capturer)
    except KeyboardInterrupt:
        pass
    finally:
        camera_capturer.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
