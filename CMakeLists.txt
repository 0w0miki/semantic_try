cmake_minimum_required(VERSION 2.8.3)
project(semantic)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

find_package(OpenCV  REQUIRED)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  darknet_ros_msgs
  orb_slam2
  message_filters
  cv_bridge 
  image_transport
)

catkin_package(
 INCLUDE_DIRS include
 LIBRARIES semantic
 CATKIN_DEPENDS roscpp darknet_ros_msgs orb_slam2
 DEPENDS OpenCV
)


include_directories(
  include ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  include
)

# add_library(semantic
#   src/semantic.cpp
# )
add_executable(semantic
  src/semantic.cpp
)
add_dependencies(semantic ${${PROJECT_NAME}_EXPORTED_TARGETS} darknet_ros_msgs_generate_messages_cpp ${catkin_EXPORTED_TARGETS})
target_link_libraries(semantic
  ${catkin_LIBRARIES}
  ${OpenCV_INCLUDE_DIRS}
  
)