// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from ros2_yolo_msgs:msg/DetectedBox.idl
// generated code does not contain a copyright notice

#ifndef ROS2_YOLO_MSGS__MSG__DETAIL__DETECTED_BOX__STRUCT_HPP_
#define ROS2_YOLO_MSGS__MSG__DETAIL__DETECTED_BOX__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__ros2_yolo_msgs__msg__DetectedBox __attribute__((deprecated))
#else
# define DEPRECATED__ros2_yolo_msgs__msg__DetectedBox __declspec(deprecated)
#endif

namespace ros2_yolo_msgs
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct DetectedBox_
{
  using Type = DetectedBox_<ContainerAllocator>;

  explicit DetectedBox_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->x1 = 0.0f;
      this->y1 = 0.0f;
      this->x2 = 0.0f;
      this->y2 = 0.0f;
      this->servo = 0l;
    }
  }

  explicit DetectedBox_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_alloc;
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->x1 = 0.0f;
      this->y1 = 0.0f;
      this->x2 = 0.0f;
      this->y2 = 0.0f;
      this->servo = 0l;
    }
  }

  // field types and members
  using _x1_type =
    float;
  _x1_type x1;
  using _y1_type =
    float;
  _y1_type y1;
  using _x2_type =
    float;
  _x2_type x2;
  using _y2_type =
    float;
  _y2_type y2;
  using _servo_type =
    int32_t;
  _servo_type servo;

  // setters for named parameter idiom
  Type & set__x1(
    const float & _arg)
  {
    this->x1 = _arg;
    return *this;
  }
  Type & set__y1(
    const float & _arg)
  {
    this->y1 = _arg;
    return *this;
  }
  Type & set__x2(
    const float & _arg)
  {
    this->x2 = _arg;
    return *this;
  }
  Type & set__y2(
    const float & _arg)
  {
    this->y2 = _arg;
    return *this;
  }
  Type & set__servo(
    const int32_t & _arg)
  {
    this->servo = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    ros2_yolo_msgs::msg::DetectedBox_<ContainerAllocator> *;
  using ConstRawPtr =
    const ros2_yolo_msgs::msg::DetectedBox_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<ros2_yolo_msgs::msg::DetectedBox_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<ros2_yolo_msgs::msg::DetectedBox_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      ros2_yolo_msgs::msg::DetectedBox_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<ros2_yolo_msgs::msg::DetectedBox_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      ros2_yolo_msgs::msg::DetectedBox_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<ros2_yolo_msgs::msg::DetectedBox_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<ros2_yolo_msgs::msg::DetectedBox_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<ros2_yolo_msgs::msg::DetectedBox_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__ros2_yolo_msgs__msg__DetectedBox
    std::shared_ptr<ros2_yolo_msgs::msg::DetectedBox_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__ros2_yolo_msgs__msg__DetectedBox
    std::shared_ptr<ros2_yolo_msgs::msg::DetectedBox_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const DetectedBox_ & other) const
  {
    if (this->x1 != other.x1) {
      return false;
    }
    if (this->y1 != other.y1) {
      return false;
    }
    if (this->x2 != other.x2) {
      return false;
    }
    if (this->y2 != other.y2) {
      return false;
    }
    if (this->servo != other.servo) {
      return false;
    }
    return true;
  }
  bool operator!=(const DetectedBox_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct DetectedBox_

// alias to use template instance with default allocator
using DetectedBox =
  ros2_yolo_msgs::msg::DetectedBox_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace ros2_yolo_msgs

#endif  // ROS2_YOLO_MSGS__MSG__DETAIL__DETECTED_BOX__STRUCT_HPP_
