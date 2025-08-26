find_package(catkin REQUIRED COMPONMENTS
roscpp
sensor_msgs
cv_bridge
image_transport
pcl_ros
pcl_conversions
darknet_ros_msgs)

include_directories
(
	${catkin_INCLUDE_DIRS}
	${PCL_INCLUDE_DIRS}
)

target_link_libraries(lidar_camera_fusion
${catkin_LIBRARIES}
${PCL_LIBRARIES}
${OpenCV_LIBRARIES}
)