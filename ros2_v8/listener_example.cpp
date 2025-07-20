#include <rclcpp/rclcpp.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>
#include <std_msgs/msg/int32.hpp>

class DetectionListener : public rclcpp::Node
{
public:
    DetectionListener() : Node("detection_listener")
    {
        // 订阅检测结果
        detection_sub_ = this->create_subscription<vision_msgs::msg::Detection2DArray>(
            "detection2d_array", 10,
            std::bind(&DetectionListener::detection_callback, this, std::placeholders::_1));
            
        // 订阅舵机状态
        servo_sub_ = this->create_subscription<std_msgs::msg::Int32>(
            "servo_state", 10,
            std::bind(&DetectionListener::servo_callback, this, std::placeholders::_1));
            
        RCLCPP_INFO(this->get_logger(), "Detection Listener 启动");
    }

private:
    void detection_callback(const vision_msgs::msg::Detection2DArray::SharedPtr msg)
    {
        RCLCPP_INFO(this->get_logger(), "收到 %zu 个检测目标", msg->detections.size());
        
        for (size_t i = 0; i < msg->detections.size(); ++i) {
            const auto& detection = msg->detections[i];
            
            // 获取边界框信息
            double center_x = detection.bbox.center.position.x;
            double center_y = detection.bbox.center.position.y;
            double size_x = detection.bbox.size_x;
            double size_y = detection.bbox.size_y;
            
            // 获取分类信息
            if (!detection.results.empty()) {
                std::string class_id = detection.results[0].hypothesis.class_id;
                double confidence = detection.results[0].hypothesis.score;
                
                std::string class_name = get_class_name(class_id);
                
                RCLCPP_INFO(this->get_logger(),
                    "目标 %zu: %s(id:%s) 位置:(%.1f, %.1f) 尺寸:(%.1fx%.1f) 置信度:%.3f",
                    i, class_name.c_str(), class_id.c_str(), 
                    center_x, center_y, size_x, size_y, confidence);
            }
        }
    }
    
    void servo_callback(const std_msgs::msg::Int32::SharedPtr msg)
    {
        int servo_state = msg->data;
        std::string desc;
        
        switch(servo_state) {
            case 0: desc = "待机"; break;
            case 1: desc = "右舵已投弹"; break;
            case 2: desc = "左舵已投弹"; break;
            case 3: desc = "投弹完成"; break;
            default: desc = "未知状态(" + std::to_string(servo_state) + ")"; break;
        }
        
        RCLCPP_INFO(this->get_logger(), "舵机状态: %s", desc.c_str());
    }
    
    std::string get_class_name(const std::string& class_id)
    {
        if (class_id == "0") return "circle";
        if (class_id == "1") return "stuffed";
        if (class_id == "2") return "H";
        return "unknown(" + class_id + ")";
    }
    
    rclcpp::Subscription<vision_msgs::msg::Detection2DArray>::SharedPtr detection_sub_;
    rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr servo_sub_;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<DetectionListener>());
    rclcpp::shutdown();
    return 0;
}
