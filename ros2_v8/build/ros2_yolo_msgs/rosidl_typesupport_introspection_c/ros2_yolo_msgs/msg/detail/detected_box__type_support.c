// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from ros2_yolo_msgs:msg/DetectedBox.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "ros2_yolo_msgs/msg/detail/detected_box__rosidl_typesupport_introspection_c.h"
#include "ros2_yolo_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "ros2_yolo_msgs/msg/detail/detected_box__functions.h"
#include "ros2_yolo_msgs/msg/detail/detected_box__struct.h"


#ifdef __cplusplus
extern "C"
{
#endif

void ros2_yolo_msgs__msg__DetectedBox__rosidl_typesupport_introspection_c__DetectedBox_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  ros2_yolo_msgs__msg__DetectedBox__init(message_memory);
}

void ros2_yolo_msgs__msg__DetectedBox__rosidl_typesupport_introspection_c__DetectedBox_fini_function(void * message_memory)
{
  ros2_yolo_msgs__msg__DetectedBox__fini(message_memory);
}

static rosidl_typesupport_introspection_c__MessageMember ros2_yolo_msgs__msg__DetectedBox__rosidl_typesupport_introspection_c__DetectedBox_message_member_array[5] = {
  {
    "x1",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(ros2_yolo_msgs__msg__DetectedBox, x1),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "y1",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(ros2_yolo_msgs__msg__DetectedBox, y1),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "x2",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(ros2_yolo_msgs__msg__DetectedBox, x2),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "y2",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(ros2_yolo_msgs__msg__DetectedBox, y2),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "servo",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_INT32,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(ros2_yolo_msgs__msg__DetectedBox, servo),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers ros2_yolo_msgs__msg__DetectedBox__rosidl_typesupport_introspection_c__DetectedBox_message_members = {
  "ros2_yolo_msgs__msg",  // message namespace
  "DetectedBox",  // message name
  5,  // number of fields
  sizeof(ros2_yolo_msgs__msg__DetectedBox),
  ros2_yolo_msgs__msg__DetectedBox__rosidl_typesupport_introspection_c__DetectedBox_message_member_array,  // message members
  ros2_yolo_msgs__msg__DetectedBox__rosidl_typesupport_introspection_c__DetectedBox_init_function,  // function to initialize message memory (memory has to be allocated)
  ros2_yolo_msgs__msg__DetectedBox__rosidl_typesupport_introspection_c__DetectedBox_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t ros2_yolo_msgs__msg__DetectedBox__rosidl_typesupport_introspection_c__DetectedBox_message_type_support_handle = {
  0,
  &ros2_yolo_msgs__msg__DetectedBox__rosidl_typesupport_introspection_c__DetectedBox_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_ros2_yolo_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, ros2_yolo_msgs, msg, DetectedBox)() {
  if (!ros2_yolo_msgs__msg__DetectedBox__rosidl_typesupport_introspection_c__DetectedBox_message_type_support_handle.typesupport_identifier) {
    ros2_yolo_msgs__msg__DetectedBox__rosidl_typesupport_introspection_c__DetectedBox_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &ros2_yolo_msgs__msg__DetectedBox__rosidl_typesupport_introspection_c__DetectedBox_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
