import os
import cv2
import numpy as np
from keras.preprocessing.image import img_to_array, ImageDataGenerator

# Khai bao bien mac dinh
TEMP = 1

#import classifier for face and eye detection
face_class = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


datagen_src = ImageDataGenerator()

# Doc du lieu
data_path = "data_process_age/test"
for dir_name in os.listdir(data_path):
    tmp_path = "{}/{}".format(data_path, dir_name)
    if not os.path.isdir(tmp_path):
        continue

    # Tao thu muc, kiem tra
    result_path = "data_train_age/test/{}".format(dir_name)
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    #Lay duong dan file
    for file_name in os.listdir(tmp_path):
        file_path = "{}/{}".format(tmp_path, file_name)
        if not os.path.isfile(file_path):
            continue

        img_load = cv2.imread(file_path)

        faces = face_class.detectMultiScale(img_load, 1.1, 3)

        for (x, y, w, h) in faces:

            if w > 350 or h > 350:
                if y-40 > 0 and y + 80 < np.shape(img_load)[1]:
                    img = cv2.resize(img_load[y-40:y + h + 80, x:x + w], (224, 224))
                elif y - 40 < 0 and y + 40 < np.shape(img_load)[1]:
                    img = cv2.resize(img_load[y:y + h + 40, x:x + w], (224, 224))
                elif y-40 > 0 and y + 80 > np.shape(img_load)[1]:
                    img = cv2.resize(img_load[y - 40 :y + h, x:x + w], (224, 224))
                else:
                    img = cv2.resize(img_load[y:y + h, x:x + w], (224, 224))

                cv2.imwrite(result_path + "/" + file_name, img)

        print(str(TEMP))
        TEMP+=1

