import numpy as np
import cv2
import tensorflow as tf

# from networks.yolo3.functions import BoundBox

detect_fn = tf.saved_model.load("cartoon/model/saved_model")

category_index = {1: {"id": 1, "name": "cartoon"}}

score_thresh = 0.5
max_detections = 30


class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, objness=None, classes=None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.objness = objness
        self.classes = classes
        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)

        return self.label

    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]

        return self.score


def get_cartoon(image, bboxes, labels, scores, thresh):  # obtencion de los puntos
    (h, w, d) = image.shape
    boxes = []
    for bbox, label, score in zip(bboxes, labels, scores):
        if score > thresh:
            xmin, ymin = int(bbox[1] * w), int(bbox[0] * h)
            xmax, ymax = int(bbox[3] * w), int(bbox[2] * h)
            box = BoundBox(xmin, ymin, xmax, ymax)

            boxes.append(box)

    return boxes


class Cartoon:
    def __init__(self):
        name = "Cartoon v1"

    def proces_cartoon(self, pathnass):
        image = cv2.imread(pathnass)
        image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
        # The model expects a batch of images, so also add an axis with `tf.newaxis`.
        input_tensor = tf.convert_to_tensor(image_np)[tf.newaxis, ...]
        detections = detect_fn(input_tensor)
        # All outputs are batches tensors.
        # Convert to numpy arrays, and take index [0] to remove the batch dimension.
        # We're only interested in the first num_detections.
        scores = detections["detection_scores"][0, :max_detections].numpy()
        bboxes = detections["detection_boxes"][0, :max_detections].numpy()
        labels = (
            detections["detection_classes"][0, :max_detections].numpy().astype(np.int64)
        )
        labels = [category_index[n]["name"] for n in labels]
        boxes = get_cartoon(image, bboxes, labels, scores, score_thresh)
        print(boxes)
        cajas = list()
        if len(boxes) != 0:
            no_cartoon = False
            for count, box in enumerate(boxes):
                coords = list()
                coords.append(int(box.xmin))
                coords.append(int(box.ymin))
                coords.append(int(box.xmax))
                coords.append(int(box.ymax))

                cajas.append(coords)
        else:
            no_cartoon = True

        return cajas, scores, labels, no_cartoon
