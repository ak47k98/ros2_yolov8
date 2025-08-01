#!/bin/bash



# 项目根目录（根据实际情况调整）
PROJECT_DIR="/home/ak47k98/PycharmProjects/ros2_v8"

# RTSP地址参数（可根据需要修改）
#RTSP_URL="rtsp://192.168.144.25:8554/main.264"

echo "=== 进入项目目录：$PROJECT_DIR ==="
cd "$PROJECT_DIR" || { echo "目录切换失败！"; exit 1; }

echo -e "\n=== 初始化Conda环境 ==="
# 根据实际conda安装路径调整（通常为~/miniconda3或~/anaconda3）
source ~/anaconda3/etc/profile.d/conda.sh
echo -e "\n=== 激活yolov8_10环境 ==="
conda activate yolov8_10 || { echo "环境激活失败！"; exit 1; }

echo -e "\n=== 设置ROS2工作区 ==="
source install/setup.bash || { echo "工作区设置失败！"; exit 1; }

echo -e "\n=== 启动YOLOv8保存节点==="
ros2 run ros_yolo save

