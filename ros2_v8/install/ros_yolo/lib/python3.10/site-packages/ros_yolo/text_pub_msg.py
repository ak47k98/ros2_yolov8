import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32
import threading


class KeyboardInputNode(Node):
    """监听终端输入的节点"""

    def __init__(self):
        super().__init__('keyboard_input_node')

        # 创建发布器
        self.state_publisher_ = self.create_publisher(Int32, 'current_state', 10)

        # 启动线程，处理键盘输入
        self.input_thread = threading.Thread(target=self.get_keyboard_input)
        self.input_thread.daemon = True
        self.input_thread.start()

    def get_keyboard_input(self):
        """监听并处理键盘输入"""
        while rclpy.ok():
            user_input = input("请输入数字（0-4）：")
            if user_input in ['0', '1', '2', '3', '4']:
                msg = Int32()
                msg.data = int(user_input)  # 将输入的字符转换为整数
                self.state_publisher_.publish(msg)
                self.get_logger().info(f"发布 current_state: {msg.data}")
            else:
                self.get_logger().info("无效输入，请输入数字 0-4")


def main(args=None):
    rclpy.init(args=args)
    keyboard_input_node = KeyboardInputNode()

    try:
        rclpy.spin(keyboard_input_node)
    except KeyboardInterrupt:
        pass
    finally:
        keyboard_input_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
