import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/wynter/wynter/ros2_yolov8/ros2_v8/install/ros_yolo'
