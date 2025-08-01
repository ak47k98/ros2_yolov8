# ros_yolo/servo_controller.py

import rclpy
from rclpy.node import Node
from mavros_msgs.srv import CommandLong
import time


class ServoController:
    def __init__(self, node: Node, namespace: str = "/mavros/"):
        self.node = node
        self.cli = node.create_client(CommandLong, namespace + "cmd/command")

        # 等待服务
        self.node.get_logger().info("等待 /mavros/cmd/command 服务...")
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().warn("等待中...")

        self.node.get_logger().info("舵机控制器初始化完成")

    def set_servo(self, servo_num: int, pwm: float):
        """底层函数：设置某个舵机PWM值"""
        req = CommandLong.Request()
        req.command = 183  # MAV_CMD_DO_SET_SERVO
        req.param1 = float(servo_num)
        req.param2 = float(pwm)
        req.param3 = 0.0
        req.param4 = 0.0
        req.param5 = 0.0
        req.param6 = 0.0
        req.param7 = 0.0
        req.confirmation = 0
        req.broadcast = False

        future = self.cli.call_async(req)

        def _callback(fut):
            try:
                resp = fut.result()
                if resp.success:
                    self.node.get_logger().info(f"[OK] 舵机{servo_num} 设置为 {pwm}")
                else:
                    self.node.get_logger().error(f"[FAIL] 舵机{servo_num} 设置失败")
            except Exception as e:
                self.node.get_logger().error(f"[EXCEPTION] 舵机{servo_num}: {e}")

        future.add_done_callback(_callback)

    def fire_servo(self, servo_id: int, delay: float = 0.5):
        """
        控制指定舵机执行一次投弹动作：
        PWM 从 1164 → 1864 → 1164
        """
        self.node.get_logger().info(f"Servo{servo_id} → 开始投弹")

        # 第一步：发射
        #self.set_servo(servo_id, 1864)
        #time.sleep(delay)

        # 第二步：收回
        #self.set_servo(servo_id, 1200)#1164
        #time.sleep(delay)

        # 第三步：复位
        self.set_servo(servo_id, 1864)
        self.node.get_logger().info(f"Servo{servo_id} → 投弹完成")
