#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from darknet_ros_msgs.msg import BoundingBoxes, BoundingBox
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO

class Yolo11Node:
    def __init__(self):
        rospy.init_node('yolov11_node', anonymous=True)

        self.bridge = CvBridge()
        self.model = YOLO("/path/to/yolov11n.pt")  # nano / tiny 模型

        self.img_sub = rospy.Subscriber("/camera/image_raw", Image, self.img_callback, queue_size=1)
        self.boxes_pub = rospy.Publisher("/yolov11n/bounding_boxes", BoundingBoxes, queue_size=1)

    def img_callback(self, msg):
        # 转换 ROS Image → OpenCV
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        # YOLO 推理
        results = self.model(cv_image)

        # 解析检测结果
        boxes_msg = BoundingBoxes()
        boxes_msg.header.stamp = rospy.Time.now()
        boxes_msg.header.frame_id = msg.header.frame_id

        for r in results[0].boxes:
            box = BoundingBox()
            box.xmin = int(r.xyxy[0][0])
            box.ymin = int(r.xyxy[0][1])
            box.xmax = int(r.xyxy[0][2])
            box.ymax = int(r.xyxy[0][3])
            box.probability = float(r.conf[0])
            box.Class = str(self.model.names[int(r.cls[0])])
            boxes_msg.bounding_boxes.append(box)

        self.boxes_pub.publish(boxes_msg)

    def spin(self):
        rospy.spin()


if __name__ == "__main__":
    node = Yolo11Node()
    node.spin()
