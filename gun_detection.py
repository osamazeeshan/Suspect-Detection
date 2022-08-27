# USAGE
# python3 gun_detection.py --image images/armas_767.jpg 

import cv2
import numpy as np
import time
import argparse

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())


# Load Yolo
net = cv2.dnn.readNet("yolov3_custom_train_3000.weights", "cfg/yolov3_custom_train.cfg")
classes = []
with open("data/yolo.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 1))

_gun_detected = False
font = cv2.FONT_HERSHEY_PLAIN

frame = cv2.imread(args["image"])
height, width, channels = frame.shape

blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

net.setInput(blob)
outs = net.forward(output_layers)

# Showing informations on the screen
class_ids = []
confidences = []
boxes = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.2:
            # Object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

            _gun_detected = True

        else:
            _gun_detected = False

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.3)

for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        _gun_detected = True
        confidence = confidences[i]
        color = colors[class_ids[i]]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 128), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + 30), (0, 0, 128), -1)
        cv2.putText(frame, label + " " + str(round(confidence, 2)), (x, y + 30), font, 3, (255,255,255), 3) 
        
if _gun_detected == True:
    print("Gun Detected!")
else:
    print("Nothing Detected!")

cv2.imshow("Image", frame)
cv2.waitKey(0)
