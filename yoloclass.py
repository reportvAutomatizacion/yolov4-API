import os
# comment out below line to enable tensorflow outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from core.functions import *
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

print('CARGANDO MODELO')
director = './checkpoints/yolov4-416'
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
model= 'yolov4'# 'yolov3 or yolov4'


class Yolo:

    def __init__(self):
        
        self.framework='tf' #(tf, tflite, trt')
        self.weights= './checkpoints/yolov4-416' #'path to weights file'
        self.size= 416# 'resize images to'
        self.tiny= True# 'yolo or yolo-tiny'
        self.model= 'yolov4'# 'yolov3 or yolov4'
        self.output= './detections/'# 'path to output folder'
        self.iou= 0.45# 'iou threshold'
        self.score= 0.75 #'score threshold
        self.count=False # count objects within images
        self.dont_show=True #'dont show image output'
        self.info=False #print info on detections
        self.crop=False #crop detections from images
        self.ocr=False #perform generic OCR on detection regions
        self.plate=False #perform license plate recognition
        

    async def yolo_v4(self, img_path):

        no_object=True

        print('Yolo v4')
        STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(self.tiny,self.model)
        input_size = self.size
        images = [img_path]
        
        # load model
        if self.framework == 'tflite':
                interpreter = tf.lite.Interpreter(model_path=self.weights)
        else:
                saved_model_loaded = tf.saved_model.load(self.weights, tags=[tag_constants.SERVING])

        # loop through images in list and run Yolov4 model on each
        for count, image_path in enumerate(images, 1):
            print('image_path',image_path)
            original_image = cv2.imread(image_path)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

            image_data = cv2.resize(original_image, (input_size, input_size))
            image_data = image_data / 255.
            
            # get image name by using split method
            image_name = image_path.split('/')[-1]
            image_name = image_name.split('.')[0]

            images_data = []
            for i in range(1):
                images_data.append(image_data)
            images_data = np.asarray(images_data).astype(np.float32)

            if self.framework == 'tflite':
                interpreter.allocate_tensors()
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()
                interpreter.set_tensor(input_details[0]['index'], images_data)
                interpreter.invoke()
                pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
                if self.model == 'yolov3' and self.tiny == True:
                    boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25, input_shape=tf.constant([input_size, input_size]))
                else:
                    boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25, input_shape=tf.constant([input_size, input_size]))
            else:
                infer = saved_model_loaded.signatures['serving_default']
                batch_data = tf.constant(images_data)
                pred_bbox = infer(batch_data)
                for key, value in pred_bbox.items():
                    boxes = value[:, :, 0:4]
                    pred_conf = value[:, :, 4:]

            # run non max suppression on detections
            boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
                boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
                scores=tf.reshape(
                    pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
                max_output_size_per_class=50,
                max_total_size=50,
                iou_threshold=self.iou,
                score_threshold=self.score
            )

            # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, xmax, ymax
            original_h, original_w, _ = original_image.shape
            bboxes = utils.format_boxes(boxes.numpy()[0], original_h, original_w)
            
            # hold all detection data in one variable
            pred_bbox = [bboxes, scores.numpy()[0], classes.numpy()[0], valid_detections.numpy()[0]]

            # read in all class names from config
            class_names = utils.read_class_names(cfg.YOLO.CLASSES)

            # by default allow all classes in .names file
            allowed_classes = list(class_names.values())
            
            # custom allowed classes (uncomment line below to allow detections for only people)
            #allowed_classes = ['person']

            # if crop flag is enabled, crop each detection and save it as new image
            if self.crop:
                crop_path = os.path.join(os.getcwd(), 'detections', 'crop', image_name)
                try:
                    os.mkdir(crop_path)
                except FileExistsError:
                    pass
                crop_objects(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB), pred_bbox, crop_path, allowed_classes)

            # if ocr flag is enabled, perform general text extraction using Tesseract OCR on object detection bounding box
            if self.ocr:
                ocr(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB), pred_bbox)

            # if count flag is enabled, perform counting of objects
            if self.count:
                # count objects found
                counted_classes = count_objects(pred_bbox, by_class = False, allowed_classes=allowed_classes)
                # loop through dict and print
                for key, value in counted_classes.items():
                    print("Number of {}s: {}".format(key, value))
                image = utils.draw_bbox(original_image, pred_bbox, self.info, counted_classes, allowed_classes=allowed_classes, read_plate = self.plate)
            else:
                image = utils.draw_bbox(original_image, pred_bbox, self.info, allowed_classes=allowed_classes, read_plate = self.plate)
            
            image = Image.fromarray(image.astype(np.uint8))
            if not self.dont_show:
                image.show()
            image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
            cv2.imwrite(self.output + 'detection' + str(count) + '.png', image)
            
            cajas=list()
            etiquetas=list()

            boxes, scores, pred_classes, num_objects = pred_bbox

            for count,box in enumerate(boxes):
                if int(box[0])!=0 or int(box[1])!=0 or int(box[2])!=0 or int(box[3]!=0):

                    coords=list()
                    coords.append(int(box[0]))
                    coords.append(int(box[1]))
                    coords.append(int(box[2]))
                    coords.append(int(box[3]))

                    no_object=False  
                    cajas.append(coords)

                    classes = read_class_names(cfg.YOLO.CLASSES)

                    class_ind = int(pred_classes[count])
                    class_name = classes[class_ind]

                    etiquetas.append(class_name)
                
            scores=list(map(float,scores))

            num_objects=int(num_objects)
            
            return cajas, scores, etiquetas, num_objects, no_object

 