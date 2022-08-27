# python3 realtime_gun_detection.py 

import cv2
import numpy as np
import time
# Load Yolo
net = cv2.dnn.readNet("yolov3_custom_train_3000.weights", "cfg/yolov3_custom_train.cfg")
classes = []
with open("data/yolo.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 1))

# Loading camera
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_PLAIN
starting_time = time.time()
frame_id = 0
frame_count = 0

while True:
    _, frame = cap.read()
    frame_id += 1
    height, width, channels = frame.shape
    cv2.putText(frame, "Press 'Esc' to exit", (950, 700), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 200), 2)

    if frame_count > 15:
        frame_count = 0
        print(frame_count)
        # Detecting objects
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

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.3)

        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = confidences[i]
                color = colors[class_ids[i]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.rectangle(frame, (x, y), (x + w, y + 30), color, -1)
                cv2.putText(frame, label + " " + str(round(confidence, 2)), (x, y + 30), font, 3, (255,255,255), 3)


    elapsed_time = time.time() - starting_time
    fps = frame_id / elapsed_time
    display_name = "Gun Detection"
    cv2.putText(frame, "FPS: " + str(round(fps, 2)), (10, 50), font, 3, (0, 0, 0), 3)
    cv2.namedWindow(display_name)        # Create a named window
    cv2.moveWindow(display_name, 50,50)  # Move it to (40,30)
    cv2.imshow(display_name, frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

    frame_count = frame_count + 1

cap.release()
cv2.destroyAllWindows()