#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml
from scipy.spatial import KDTree

import os
from PIL import Image as im


STATE_COUNT_THRESHOLD = 3

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.waypoints_2d = None
        self.waypoints_tree = None
        self.has_image = False
        self.camera_image = None
        self.light_state = TrafficLight.RED
        # self.image_cnt = 0         # used for naming images saved --zll
        # self.image_filepath = '/home/workspace/CarND-Capstone/ros/images/'
        # self.image_filepath = '/home/udacity/CarND-Capstone/ros/images/'
        self.image_filepath = os.path.abspath('../..') + '/images/'
        self.lights = None

        self.LOCAL_HIGHT = 0
        self.LOCAL_WIDTH = 0
        self.HIGHT = 500
        self.WIDTH = 800

        self.diff = 100

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()

        self.light_classifier = TLClassifier()
        rospy.logwarn(self.light_classifier.SSD_GRAPH_FILE)

        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        # rospy.spin()
        # zhangliangliang added
        # change image callback method to loop
        # to test the process_traffic_lights
        # and the Waypoints Updater Node
        self.loop()

    def loop(self):
        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            # if self.pose and self.waypoints and self.lights and self.camera_image:
            if self.pose and self.waypoints and self.camera_image:
                self.publish_waypoints()
            rate.sleep()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints
        if not self.waypoints_2d:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in
                                 waypoints.waypoints]
            self.waypoints_tree = KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg
        # cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")
        # cv2.imwrite(self.image_filepath + "image_{}.jpg".format(self.image_cnt), cv_image)
        # self.image_cnt += 1

    def publish_waypoints(self):

        light_wp, state = self.process_traffic_lights()
        # rospy.logwarn("Closest light wp: {0} \n And light state: {1}".format(light_wp, state))

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

    def get_closest_waypoint(self, x, y):
        """Identifies the closest path waypoint to the given position https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        # TODO implement
        closest_idx = self.waypoints_tree.query([x, y], 1)[1]
        return closest_idx

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        # For testing, just return the light state
        # return light.state

        # second step:
        # detect traffic light, and save it
        if self.has_image:
            cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            cv_image = cv_image[self.LOCAL_HIGHT: self.LOCAL_HIGHT + self.HIGHT, self.LOCAL_WIDTH: self.LOCAL_WIDTH + self.WIDTH]
            image = im.fromarray(cv_image)
            self.light_state = TrafficLight.RED
            traffic_light = self.light_classifier.get_location(image)
            if traffic_light:
                [top, left, bottom, right] = traffic_light
                # rospy.logwarn("[top:{0} bottom:{1} left:{2} right:{3}]".format(top, bottom, left, right))
                cropped = cv_image[top: bottom, left: right]
                # rospy.logwarn("cropped_image: {0}".format(cropped.shape))
                # cv2.imwrite(self.image_filepath + "image_{}.jpg".format(self.image_cnt), cropped)
                # self.image_cnt += 1
                self.light_state = self.light_classifier.get_classification(cropped)
                rospy.logwarn("light state: {0}".format(self.light_state))
                # rospy.logwarn("light state: {0}, equal {1}".format(self.light_state, light.state))

            self.has_image = False

        # but we still use light.state
        # return light.state
        return self.light_state

        # if(not self.has_image):
        #     self.prev_light_loc = None
        #     return False
        #
        # cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")
        #
        # # Get classification
        # return self.light_classifier.get_classification(cv_image)

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        closet_light_i = None
        line_wp_idx = None

        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        if self.pose:
            car_wp_idx = self.get_closest_waypoint(self.pose.pose.position.x, self.pose.pose.position.y)

            # TODO find the closest visible traffic light (if one exists)
            # diff = len(self.waypoints.waypoints)
            diff = self.diff
            for i, line in enumerate(stop_line_positions):
                # Get stop line waypoint index
                # line = stop_line_positions[i]
                temp_wp_idx = self.get_closest_waypoint(line[0], line[1])
                # Find closest stop line waypoint index
                d = temp_wp_idx - car_wp_idx
                if d >= 0 and d < diff:
                    diff = d
                    closet_light_i = i
                    line_wp_idx = temp_wp_idx

        # rospy.logwarn("Closest light wp: {0} \n And light state: {1}".format(line_wp_idx, closet_light_i))
        # if closet_light:
        if line_wp_idx:
            # -------------------------------------------------------------
            # this is the only place to use self.lights function
            # light = self.lights[closet_light_i]
            light = None
            state = self.get_light_state(light)
            # -------------------------------------------------------------
            return line_wp_idx, state

        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
