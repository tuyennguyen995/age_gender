import os
import random
import shutil
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

# Khai bao bien mac dinh
TEMP0 = 0
TEMP1 = 0
TEMP2 = 0

# Doc du lieu
data_path = "dataset/gender"
for dir_name in os.listdir(data_path):
    tmp_path = "{}/{}".format(data_path, dir_name)
    if not os.path.isdir(tmp_path):
        continue

    # Tao thu muc, kiem tra
    result_path_train = "data_process_gender/train/{}".format(dir_name)
    if not os.path.exists(result_path_train):
        os.makedirs(result_path_train)

    result_path_validation = "data_process_gender/validation/{}".format(dir_name)
    if not os.path.exists(result_path_validation):
        os.makedirs(result_path_validation)

    result_path_test = "data_process_gender/test/{}".format(dir_name)
    if not os.path.exists(result_path_test):
        os.makedirs(result_path_test)

    #Lấy đường dẫn file
    for file_name in os.listdir(tmp_path):
        file_path = "{}/{}".format(tmp_path, file_name)
        if not os.path.isfile(file_path):
            continue

        #Chuyển file ảnh
        x = random.randint(1, 100)
        
        if TEMP1 < 400 and x <= 20:
            shutil.copy(file_path, result_path_validation)
            TEMP1 += 1
        elif TEMP2 < 200 and (x > 20 and x <= 35) :
            shutil.copy(file_path, result_path_test)
            TEMP2 += 1
        else:
            shutil.copy(file_path, result_path_train)
            
        TEMP0 += 1
        print(str(TEMP0))
    TEMP1 = 0
    TEMP2 = 0
