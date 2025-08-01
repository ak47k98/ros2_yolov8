# ros_yolo/test_servo_node.py

import rclpy
from rclpy.node import Node
from ros_yolo.servo_controller import ServoController


class ServoTestNode(Node):
    def __init__(self):
        super().__init__("servo_test_node")

        self.servo_ctrl = ServoController(self, "/mavros/")

        # 直接测试两个舵机发射动作
        self.timer = self.create_timer(3.0, self.test_sequence)
        self.step = 0

    def test_sequence(self):
        if self.step == 0:
            self.get_logger().info("测试 Servo11（右舵）发射")
            self.servo_ctrl.fire_servo(11)
        elif self.step == 1:
            self.get_logger().info("测试 Servo12（左舵）发射")
            self.servo_ctrl.fire_servo(12)
        else:
            self.get_logger().info("测试完成，关闭节点")
            rclpy.shutdown()
        self.step += 1


def main(args=None):
    rclpy.init(args=args)
    node = ServoTestNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
