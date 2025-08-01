#!/bin/bash

# pip install MAVproxy
sudo chmod 777 /dev/ttyACM0
# 启动MAVProxy作为主要中继
gnome-terminal -t "mavproxy" -x bash -c "
source ~/anaconda3/etc/profile.d/conda.sh
conda activate droneplanner
# MAVProxy连接SITL并转发到多个端口
mavproxy.py \
    --master='/dev/ttyACM0' \
    --out=udp:127.0.0.1:14551 \
    --out=udp:127.0.0.1:14552 \
    --console --moddebug 3;
exec bash;
"
# 等待MAVProxy启动
sleep 3


# 启动MAVROS连接到MAVProxy转发的端口
gnome-terminal -t "mavros" -x bash -c "
# cd /home/linhao/code/ros2/dxy_apm_ws;
source /opt/ros/jazzy/setup.bash;

ros2 launch mavros apm.launch \
    fcu_url:=udp://127.0.0.1:14551@14555 \

exec bash;
"
