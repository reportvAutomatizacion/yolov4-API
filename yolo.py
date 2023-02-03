import gc
import logging
import os

import cv2
import numpy as np
import tensorflow as tf
from keras import backend as K
from PIL import Image
from tensorflow.compat.v1 import ConfigProto, InteractiveSession
from tensorflow.python.saved_model import tag_constants

import core.utils as utils
from core.functions import *

logging.basicConfig(level=logging.INFO)

# comment out below line to enable tensorflow outputs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

physical_devices = tf.config.experimental.list_physical_devices("GPU")
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


class Yolo:
    def __init__(self):
        self.framework = "tf"  # (tf, tflite, trt')
        self.weights = "./checkpoints/yolov4-416"  #'path to weights file'
        self.size = 416  # 'resize images to'
        # self.model= 'yolov4'# 'yolov3 or yolov4'
        self.output = "/NASReporTV/detections/"  # 'path to output folder'
        self.iou = 0.45  # 'iou threshold'
        self.score = 0.75  #'score threshold
        self.count = False  # count objects within images
        self.dont_show = True  #'dont show image output'
        self.info = False  # print info on detections
        self.crop = False  # crop detections from images
        self.ocr = False  # perform generic OCR on detection regions
        self.plate = False  # perform license plate recognition
        self.volume_path = ""  # "/NASReporTV/"
        self.model_loaded = self.load_model()

    def load_model(self):
        logging.info("Loading yolov4 model....")
        return tf.saved_model.load(self.weights, tags=[tag_constants.SERVING])

    def preprocess_image(self, image_path):
        logging.info("Processing the image......")
        images_data = []
        original_image = cv2.imread(image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        input_size = self.size
        image_data = cv2.resize(original_image, (input_size, input_size))
        image_data = image_data / 255.0
        images_data.append(image_data)
        images_data = np.asarray(images_data).astype(np.float32)

        return images_data, original_image

    def get_model_predictions(self, images_data):
        logging.info("Getting detections.....")
        infer = self.model_loaded.signatures["serving_default"]
        batch_data = tf.constant(images_data)
        pred_bbox = infer(batch_data)

        return pred_bbox

    def get_detections(self, img_path):
        no_object = True
        complete_path = self.volume_path + img_path
        images_data, original_image = self.preprocess_image(complete_path)
        # get image name by using split method
        image_name = complete_path.split("/")[-1]
        image_name = image_name.split(".")[0]

        # Model inference
        pred_bbox = self.get_model_predictions(images_data)

        for _, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]

        # run non max suppression on detections
        (
            boxes,
            scores,
            classes,
            valid_detections,
        ) = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])
            ),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=self.iou,
            score_threshold=self.score,
        )

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, xmax, ymax
        original_h, original_w, _ = original_image.shape
        bboxes = utils.format_boxes(boxes.numpy()[0], original_h, original_w)

        print(valid_detections)

        # hold all detection data in one variable
        pred_bbox = [
            bboxes,
            scores.numpy()[0],
            classes.numpy()[0],
            valid_detections.numpy()[0],
        ]

        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # by default allow all classes in .names file
        allowed_classes = list(class_names.values())

        # custom allowed classes (uncomment line below to allow detections for only people)
        # allowed_classes = ['person']

        # Draw predicted boxes
        image = utils.draw_bbox(
            original_image,
            pred_bbox,
            self.info,
            allowed_classes=allowed_classes,
            read_plate=self.plate,
        )

        # Save binary image
        image = Image.fromarray(image.astype(np.uint8))
        image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
        cv2.imwrite(self.output + image_name + ".png", image)

        boxes, scores, pred_classes, num_objects = pred_bbox

        # Take detections with score > 0
        score_array = np.array(scores)
        indices = np.where(score_array > 0)[0]
        scores = scores[indices]
        boxes = boxes[indices].tolist()
        pred_classes = pred_classes[indices]
        classes = read_class_names(cfg.YOLO.CLASSES)
        classes_names = [classes[int(number)] for number in pred_classes]

        scores = list(map(float, scores))

        num_objects = int(num_objects)

        K.clear_session()
        gc.collect()

        return boxes, scores, classes_names, num_objects, no_object

    def run_another_utils(self, image_name, original_image, pred_bbox, allowed_classes):
        # if crop flag is enabled, crop each detection and save it as new image
        if self.crop:
            crop_path = os.path.join(os.getcwd(), "detections", "crop", image_name)
            try:
                os.mkdir(crop_path)
            except FileExistsError:
                pass
            crop_objects(
                cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB),
                pred_bbox,
                crop_path,
                allowed_classes,
            )

        # if ocr flag is enabled, perform general text extraction using Tesseract OCR on object detection bounding box
        if self.ocr:
            ocr(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB), pred_bbox)

        # if count flag is enabled, perform counting of objects
        if self.count:
            # count objects found
            counted_classes = count_objects(
                pred_bbox, by_class=False, allowed_classes=allowed_classes
            )
            # loop through dict and print
            for key, value in counted_classes.items():
                print("Number of {}s: {}".format(key, value))
            image = utils.draw_bbox(
                original_image,
                pred_bbox,
                self.info,
                counted_classes,
                allowed_classes=allowed_classes,
                read_plate=self.plate,
            )
