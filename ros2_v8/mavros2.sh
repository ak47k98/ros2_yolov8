#!/bin/bash

sudo chmod 777 /dev/ttyACM0
#ros2 launch mavros apm.launch fcu_url:='udp://192.168.7.106:14550@14555'
ros2 launch mavros apm.launch fcu_url:='/dev/ttyACM0'

