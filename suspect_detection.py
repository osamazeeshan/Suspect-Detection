# python3 suspect_detection.py --encodings encoding.pickle

from imutils.video import VideoStream
import face_recognition
import argparse
import imutils
import pickle
import time
import cv2
import numpy as np

from threading import Thread

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True,
	help="path to serialized db of facial encodings")
ap.add_argument("-o", "--output", type=str,
	help="path to output video")
ap.add_argument("-y", "--display", type=int, default=1,
	help="whether or not to display output frame to screen")
ap.add_argument("-d", "--detection-method", type=str, default="cnn",
	help="face detection model to use: either `hog` or `cnn`")
ap.add_argument('--graph', default='output_graph.pb', type=str,
    help='Absolute path to graph file (.pb)')
ap.add_argument('--labels', default='labels.txt', type=str,
    help='Absolute path to labels file (.txt)')
args = vars(ap.parse_args())

# load the known faces and embeddings
print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())

# Load Yolo
net = cv2.dnn.readNet("yolov3_custom_train_3000.weights", "cfg/yolov3_custom_train.cfg")
classes = []
with open("data/yolo.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 1))

gun_font = cv2.FONT_HERSHEY_PLAIN

class VideoStreamWidget(object):

    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src)
        # Start the thread to read frames from the video stream
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        fram_count = 0
        y = 0
        top = 0
        right = 0
        bottom = 0
        left = 0
        name = ''
        gun_left = 0 
        gun_top = 0 
        gun_width = 0 
        gun_height = 0
        _suspect_person = False
        _suspect_gun = False
        _threshold_value = 30
        output_layer = 'final_result:0'
        input_layer = 'Mul:0'
        num_top_predictions = 5

        while True:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()
                cv2.putText(self.frame, "Press 'Esc' to exit", (950, 700), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 200), 2)
            
                if fram_count > 15 :
                    rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                    rgb = imutils.resize(self.frame, width=200)
                    r = self.frame.shape[1] / float(rgb.shape[1])
                    boxes = face_recognition.face_locations(rgb, model=args["detection_method"])
                    encodings = face_recognition.face_encodings(rgb, boxes)
                    names = []
                    fram_count = 0

                    for encoding in encodings:
                        matches = face_recognition.compare_faces(data["encodings"],
                        encodings[0])
                        name = "Unknown"

                        if True in matches:
                            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                            counts = {}

                            for i in matchedIdxs:
                                name = data["names"][i]
                                counts[name] = counts.get(name, 0) + 1

                            name = max(counts, key=counts.get)
                            threshold_count = counts.get(name, 0)
                            # print (threshold_count)
                            if threshold_count > _threshold_value:
                                _suspect_person = False
                            else:
                                _suspect_person = True

                        names.append(name)
                    
                    for ((top, right, bottom, left), name) in zip(boxes, names):

                        top = int(top * r)
                        right = int(right * r)
                        bottom = int(bottom * r)
                        left = int(left * r)

                        if _suspect_person == True:
                            
                            name = 'Suspect Found!!'
                            cv2.rectangle(self.frame, (left, top), (right, bottom), (0, 0, 128), 3)
                            cv2.putText(self.frame, name, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 128), 9)
                        else:
                            cv2.rectangle(self.frame, (left, top), (right, bottom), (0, 255, 0), 3)
                            y = top - 15 if top - 15 > 15 else top + 15
                            cv2.putText(self.frame, name, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.75, (0, 255, 0), 4)
                
                    height, width, channels = self.frame.shape
                     # Detecting objects
                    blob = cv2.dnn.blobFromImage(self.frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

                    net.setInput(blob)
                    outs = net.forward(output_layers)
                    _suspect_gun = False

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
                            _suspect_gun = True
                            gun_left, gun_top, gun_width, gun_height = boxes[i]
                            gun_label = str(classes[class_ids[i]])
                            confidence = confidences[i]
                            color = colors[class_ids[i]]
                            cv2.rectangle(self.frame, (gun_left, gun_top), (gun_left + gun_width, gun_top + gun_height), (0, 0, 128), 3)
                            cv2.rectangle(self.frame, (gun_left, gun_top), (gun_left + gun_width, gun_top + 30), (0, 0, 128), -1)
                            cv2.putText(self.frame, gun_label + " " + str(round(confidence, 2)), (gun_left, gun_top + 30), gun_font, 3, (255,255,255), 3)

                else:
                    if _suspect_person == True:
                        name = 'Suspect Found!!'
                        cv2.rectangle(self.frame, (left, top), (right, bottom), (0, 0, 128), 3)
                        cv2.putText(self.frame, name, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 128), 9)
                    
                    else:
                        cv2.rectangle(self.frame, (left, top), (right, bottom), (0, 255, 0), 3)
                        y = top - 15 if top - 15 > 15 else top + 15
                        cv2.putText(self.frame, name, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.75, (0, 255, 0), 4)

                    if _suspect_gun == True:
                        cv2.rectangle(self.frame, (gun_left, gun_top), (gun_left + gun_width, gun_top + gun_height), (0, 0, 128), 3)
                        cv2.rectangle(self.frame, (gun_left, gun_top), (gun_left + gun_width, gun_top + 30), (0, 0, 128), -1)
                        cv2.putText(self.frame, gun_label + " " + str(round(confidence, 2)), (gun_left, gun_top + 30), gun_font, 3, (255,255,255), 3)
                
                fram_count = fram_count + 1

            time.sleep(.01)

    def show_frame(self):
        # Display frames in main program
        displayName = "Face Recognition"
        cv2.namedWindow(displayName)        # Create a named window
        cv2.moveWindow(displayName, 50,50)  # Move it to (40,30)
        cv2.imshow(displayName, self.frame)
        key = cv2.waitKey(1)
        if key == 27:
            self.capture.release()
            cv2.destroyAllWindows()
            exit(1)

if __name__ == '__main__':
    video_stream_widget = VideoStreamWidget()
    while True:
        try:
            video_stream_widget.show_frame()
        except AttributeError:
            pass