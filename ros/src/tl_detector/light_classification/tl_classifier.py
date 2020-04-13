
import os
import cv2
import tensorflow as tf
import numpy as np
from styx_msgs.msg import TrafficLight
from keras.preprocessing import image
from keras.models import Sequential, load_model
from keras.layers.convolutional import Conv2D
from keras.layers.core import Dense, Activation, Flatten, Dropout


class TLClassifier(object):

    # the size of training images
    IMAGE_HEIGHT = 32
    IMAGE_WIDTH = 32
    IMAGE_CHANNEL = 3
    IMAGE_CLASSIFY = 3

    def __init__(self):
        # TODO load classifier
        # Frozen inference graph files. NOTE: change the path to where you saved the models.
        # self.SSD_GRAPH_FILE = 'ssd_mobilenet_v1_coco_11_06_2017/frozen_inference_graph.pb'
        # self.SSD_GRAPH_FILE = '/home/udacity/CarND-Capstone/ros/src/tl_detector/light_classification/' \
        #                       'ssd_mobilenet_v1_coco_11_06_2017/frozen_inference_graph.pb'
        self.SSD_GRAPH_FILE = os.path.abspath('.') + '/light_classification/ssd_mobilenet_v1_coco_11_06_2017/frozen_inference_graph.pb'

        # load graph
        self.detection_graph = self.load_graph(self.SSD_GRAPH_FILE)

        # The input placeholder for the image.
        # `get_tensor_by_name` returns the Tensor with the associated name in the Graph.
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')

        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')

        # The classification of the object (integer id).
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')

        self.confidence_cutoff = 0.5

        # class id of traffic light in ssd
        self.TRAFFIC_LIGHT_CLASS_ID = 10

        self.model_path = os.path.abspath('.') + '/light_classification/traffic_lights_classfy.h5'
        self.model = load_model(self.model_path)

    @staticmethod
    def load_graph(graph_file):
        """Loads a frozen inference graph"""
        graph = tf.Graph()
        with graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(graph_file, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        return graph

    def get_location(self, image):
        image_np = np.expand_dims(np.asarray(image, dtype=np.uint8), 0)

        with tf.Session(graph=self.detection_graph) as sess:
            # Actual detection.
            (boxes, scores, classes) = sess.run([self.detection_boxes, self.detection_scores, self.detection_classes],
                                                feed_dict={self.image_tensor: image_np})

        # Remove unnecessary dimensions
        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes)

        # Filter boxes with a confidence score less than `confidence_cutoff`
        boxes, scores, classes = self.filter_boxes(self.confidence_cutoff, boxes, scores, classes)

        # The current box coordinates are normalized to a range between 0 and 1.
        # This converts the coordinates actual location on the image.
        width, height = image.size
        box_coords = self.to_image_coords(boxes, height, width)

        traffic_light_box = self.has_traffic_light(box_coords, classes)
        # if return None, no traffic light found
        # else return the traffic light coord
        return traffic_light_box

    def has_traffic_light(self, box_coords, classes):
        """
        find the traffic light from the classes, and return the box coord
        if there are more than one traffic light, return the fist one box coord
        :param box_coords:
        :param classes:
        :return:
        """
        classes_list = classes.tolist()
        index = classes_list.index(self.TRAFFIC_LIGHT_CLASS_ID) if self.TRAFFIC_LIGHT_CLASS_ID in classes_list else -1
        if index == -1:
            return None
        else:
            top, left, bot, right = box_coords[index, ...]
            return [int(top), int(left), int(bot), int(right)]

    def to_image_coords(self, boxes, height, width):
        """
        The original box coordinate output is normalized, i.e [0, 1].

        This converts it back to the original coordinate based on the image
        size.
        """
        box_coords = np.zeros_like(boxes)
        box_coords[:, 0] = boxes[:, 0] * height
        box_coords[:, 1] = boxes[:, 1] * width
        box_coords[:, 2] = boxes[:, 2] * height
        box_coords[:, 3] = boxes[:, 3] * width

        return box_coords

    def filter_boxes(self, min_score, boxes, scores, classes):
        """Return boxes with a confidence >= `min_score`"""
        n = len(classes)
        idxs = []
        for i in range(n):
            if scores[i] >= min_score:
                idxs.append(i)

        filtered_boxes = boxes[idxs, ...]
        filtered_scores = scores[idxs, ...]
        filtered_classes = classes[idxs, ...]
        return filtered_boxes, filtered_scores, filtered_classes

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        # TODO implement light color prediction
        image_cb = self.image_preprocess(image)
        resault = self.model.predict(image_cb)
        resault = np.argmax(resault)
        # return resault
        return resault

    def image_preprocess(self, cropped):

        im_test = cv2.resize(cropped, (self.IMAGE_WIDTH, self.IMAGE_HEIGHT))
        im_test = im_test / 255.0 - 0.5

        im_test = image.img_to_array(im_test)
        im_test = np.expand_dims(im_test, axis=0)

        return im_test

