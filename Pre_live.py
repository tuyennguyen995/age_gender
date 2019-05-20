import sys
import os
import cv2
import numpy as np
import tensorflow as tf

from tkinter import filedialog
from tkinter import *

from keras.preprocessing.image import img_to_array
from keras.models import model_from_json

# Khai bao bien mac dinh
KEY_ESCAPE = 27
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720


# Phat hien khuon mat
def detect_face_open_cv_dnn(net, frame, conf_threshold=0.5):
    frame_opencv_dnn = frame.copy()
    frame_height = frame_opencv_dnn.shape[0]
    frame_width = frame_opencv_dnn.shape[1]
    blob = cv2.dnn.blobFromImage(frame_opencv_dnn, 1.0, (300, 300), [104, 117, 123], False, False)

    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    ajust = 0.17
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frame_width)
            y1 = int(detections[0, 0, i, 4] * frame_height)
            x2 = int(detections[0, 0, i, 5] * frame_width)
            y2 = int(detections[0, 0, i, 6] * frame_height)
            ajust_x = int((x2 - x1) * ajust)
            ajust_y = int((y2 - y1) * ajust)
            x1 -= ajust_x
            x2 += ajust_x
            y1 -= int(ajust_y * 1.5)
            y2 += int(ajust_y * 0.5)
            bboxes.append([x1, y1, x2, y2])
            cv2.rectangle(frame_opencv_dnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frame_height / 350)), 8)
    return frame_opencv_dnn, bboxes


if __name__ == "__main__":

    # Load model age
    json_file = open("models/train/age4/age_model.json", "r")
    json_string = json_file.read()
    json_file.close()
    model = model_from_json(json_string)
    model.load_weights("models/train/age4/weights.26.h5")
    names = ["0-2", "15-20", "25-32", "38-43", "4-6", "48-53", "60-100", "8-12"]
    # Load model gender
    json_file1 = open("models/train/gender3/gender_model.json", "r")
    json_string1 = json_file1.read()
    json_file1.close()
    model1 = model_from_json(json_string1)
    model1.load_weights("models/train/gender3/weights.28.h5")
    names1 = ["Nam", "Nu"]

    # Khoi tao moduls phat hien khuon mat
    model_file = "models/face/opencv_face_detector_uint8.pb"
    config_file = "models/face/opencv_face_detector.pbtxt"
    net = cv2.dnn.readNetFromTensorflow(model_file, config_file)

# print (model.summary())
# Nguon du lieu mac dinh la webcam
    source = 0
    if len(sys.argv) > 1:
        source = sys.argv[1]

    # Khoi tao stream video
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DEFAULT_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DEFAULT_HEIGHT)

    while True:
        has_frame, frame = cap.read()
        frame = cv2.flip(frame, 1)  # Lật ảnh
        if not has_frame:
            break

        out_opencv_dnn, bboxes = detect_face_open_cv_dnn(net, frame)

        for x1, y1, x2, y2 in bboxes:
            try:
                #Lấy khung ảnh
                a = int((x2 - x1) * 0.2)
                b = a * 2
                dd2 = ((y2 - y1) - (x2 - x1)) + int(a * 0.65)
                dd1 = int(dd2 / 2) + a
                img = cv2.resize(frame[y1 - a:y2 + b, x1 - dd1:x2 + dd2], (300, 300))

                # Tiền sử lý
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img_arr = img_to_array(img) / 255
                img_arr = img_arr.reshape((1,) + img_arr.shape)

                # Dự đoán
                prediction = model.predict(img_arr)[0]
                prediction1 = model1.predict(img_arr)[0]
                pos = np.argmax(prediction)
                pos1 = np.argmax(prediction1)

                # Tính độ chính xác
                max_percent = np.amax(prediction) * 100
                max_percent1 = np.amax(prediction1) * 100
                if max_percent < 50.0:
                    continue
                if max_percent1 < 50.0:
                    continue

                # Tính tỷ lệ khung Box
                face_width = (x2 - x1) / 260
                box_width = int(face_width * 20)
                marin = int(face_width * 10)

                # Hiển thị khung Box
                cv2.rectangle(out_opencv_dnn, (x1 + marin, y1 - box_width), (x2 - marin, y1), (0, 255, 0), box_width)
                cv2.putText(out_opencv_dnn, "{} - ({})".format(names1[pos1], names[pos]), (x1, y1),
                            cv2.FONT_HERSHEY_SIMPLEX, face_width, (255, 255, 255), 1, cv2.LINE_4)

            except Exception as e:
                print(x1, y1, x2, y2)


        cv2.imshow("Age-Gender Detection", out_opencv_dnn)

        # Bat su kien phim
        key = cv2.waitKey(10)
        if key == KEY_ESCAPE:
            break

cv2.destroyAllWindows()
cap.release()


