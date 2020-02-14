import cv2
import torch
import numpy as np
import os_cnnField as field
from pydarknet import Detector, Image

data_path = '/home/ege/Desktop/CNNField/data/traffic/out'
data_extension = '.jpg'
yolo_net = Detector(bytes("cfg/yolov3.cfg", encoding="utf-8"), \
                    bytes("weights/yolov3.weights", encoding="utf-8"), 0, \
                    bytes("cfg/coco.data", encoding="utf-8"))


def ask_yolo(index):  # tested
    file = data_path + str(index) + data_extension
    img = cv2.imread(file)
    img_darknet = Image(img)
    results = yolo_net.detect(img_darknet)
    return {r[0]: r[1] for r in results}  # list of object-confidence-coordinate triplets


def get_image(index):  # tested
    img = cv2.imread(data_path + str(index) + data_extension)
    img = np.transpose(img, (2, 0, 1))
    img_tensor = torch.from_numpy(img).unsqueeze(0).type('torch.FloatTensor')
    return img_tensor/255.0


cf = field.cnnField()

for i in range(30):
    print("Iteration: "+str(i))
    img = get_image(i+1)
    supervision = ask_yolo(i+1)
    cf.updateField(img, supervision)

print("Correct, False, Yolo:")
print(cf.correct_counter, cf.false_counter, cf.yolo_counter)
