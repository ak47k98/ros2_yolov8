#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection2DArray
from std_msgs.msg import Int32

class DetectionListener(Node):
    def __init__(self):
        super().__init__('detection_listener')
        
        # 订阅检测结果
        self.detection_sub = self.create_subscription(
            Detection2DArray,
            'detection2d_array',
            self.detection_callback,
            10
        )
        
        # 订阅舵机状态
        self.servo_sub = self.create_subscription(
            Int32,
            'servo_state',
            self.servo_callback,
            10
        )
        
        self.get_logger().info("Detection Listener 启动")

    def detection_callback(self, msg: Detection2DArray):
        """处理检测结果消息"""
        self.get_logger().info(f"收到 {len(msg.detections)} 个检测目标")
        
        for i, detection in enumerate(msg.detections):
            # 获取边界框信息
            center_x = detection.bbox.center.position.x
            center_y = detection.bbox.center.position.y
            size_x = detection.bbox.size_x
            size_y = detection.bbox.size_y
            theta = detection.bbox.center.theta
            
            # 获取分类信息
            if detection.results:
                class_id = detection.results[0].hypothesis.class_id
                confidence = detection.results[0].hypothesis.score
                
                # 根据统一类别ID解析类别名称
                class_name = self.get_class_name(class_id)
                
                self.get_logger().info(
                    f"目标 {i}: {class_name}(id:{class_id}) "
                    f"位置:({center_x:.1f}, {center_y:.1f}) "
                    f"尺寸:({size_x:.1f}x{size_y:.1f}) "
                    f"置信度:{confidence:.3f}"
                )

    def servo_callback(self, msg: Int32):
        """处理舵机状态消息"""
        servo_state = msg.data
        state_desc = {
            0: "待机",
            1: "右舵已投弹", 
            2: "左舵已投弹",
            3: "投弹完成"
        }
        
        desc = state_desc.get(servo_state, f"未知状态({servo_state})")
        self.get_logger().info(f"舵机状态: {desc}")

    def get_class_name(self, class_id: str) -> str:
        """根据统一类别ID获取类别名称"""
        class_mapping = {
            '0': 'circle',    # 圆形目标
            '1': 'stuffed',   # 投掷物目标  
            '2': 'H',         # H型降落标识
        }
        return class_mapping.get(class_id, f"unknown({class_id})")

def main(args=None):
    rclpy.init(args=args)
    node = DetectionListener()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
