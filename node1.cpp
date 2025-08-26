#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <darknet_ros_msgs/BoundingBoxes.h>

#include "structIO.hpp"
#include "dataStructures.h"

using namespace std;
class LidarCameraFusion
{
public:
	LidarCameraFusion(ros::NodeHandle& nh)
	{
		pc_sub_ = nh.subscribe("/points_raw", 1, &LidarCameraFusion::pcCallback, this);
		img_sub_ = nh.subscribe("/camera/image_raw", 1, &LidarCameraFusion::imgCallback, this);
		yolo_sub_ = nh.subscribe("/darknet_ros/bounding_boxes", &LidarCameraFusion::yoloCallback, this);

	}
private:
	ros::Subscriber pc_sub_;
	ros::Subscriber img_sub_;
	ros::Subscriber yolo_sub_;

	cv::Mat last_image_;
	std::vector<LidarPoint> last_lidar_points_;
	darknet_ros_msgs::BoundingBoxesConstPtr last_yolo_boxes_;

	void pcCallback(const sensor_msgs::PointCloud2ConstPtr& msg)
	{
		pcl::PointCloud<pcl::PointXYZ> pcl_cloud;
		pcl::fromROSMsg(*msg, pcl_cloud);

		std::vector<LidarPoint> lidarPoints;
		lidarPoints.reserve(pcl_cloud.points.size());

		for (const auto& pt : pcl_cloud.points)
		{
			if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z))
				continue;
			LidarPoint lp;
			lp.x = pt.x;
			lp.y = pt.y;
			lp.z = pt.z;
			lidarPoints.push_back(lp);
		}
		ROS_INFO("接收到点云:%lu个点", lidarPoints.size());
		last_lidar_points_ = lidarPoints;
		processFusion();

	}

	void imgCallback(const sensor_msgs::ImageConstPtr& msg)
	{
		try
		{
			cv_bridge::CvImageConstPtr cv_ptr;
			cv_ptr = cv_bridge::toCvShare(msg, "bgr8");
			last_image_ = cv_ptr->image.clone();

			ROS_INFO("接收到图像:%d x%d", last_image_.cols, last_image_.rows);
			processFusion();
		}
		catch (cv_bridge::Exception& e)
		{
			ROS_ERROR("cv_bridge 异常: %s", e.what());
		}
	}

	yoloCallback(const darknet_ros_msgs::BoundingBoxesConstPtr& msg)
	{
		last_yolo_boxes_ = msg;
		processFusion();
	}

	void processFusion()
	{
		if (last_image_.empty() || last_lidar_points_.empty() || !last_yolo_boxes_)
			return;

		std::vector<BoundingBox> boundingBoxes;
		int id = 0;
		for (const auto& box : last_yolo_boxes_->bounding_boxes)
		{
			BoundingBox bb;
			bb.boxID = id++;
			bb.ROI.x = box.xmin;
			bb.ROI.y = box.ymin;
			bb.ROI.width = box.xmax - box.xmin;
			bb.ROI.height = box.ymax - box.ymin;
			bb.classID = 0;
			boundingBoxes.push_back(bb);
		}

		clusterLidarWithROI(boundingBoxes, last_lidar_points_);

		for (auto& box : boundingBoxes)
		{
			if (box.lidarPoints.size() > 0)
			{
				showLidarTopview(box.lidarPoints, cv::Size(10.0, 25.0), cv::Size(1000, 2000));
				break;
			}
		}
	}
};

int main(int argc, char** argv)
{
	ros::init(argc, argv, "lidar_camera_fusion");
	ros::NodeHandle nh;

	LidarCameraFusion node(nh);
	ros::spin();
	return 0;
}