import os
from shutil import copyfile


img_path = "yolo_training/data/statue/"
labels_path = "yolo_training/data/museum/labels/"

suffix = " (copia 5)"

for img in os.listdir(img_path):
    title, ext = os.path.splitext(img)
    label_name = title + ".txt"
    for label in os.listdir(labels_path):
        if label == label_name:
            copyfile("{}{}.txt".format(labels_path, title), "{}{}{}.txt".format(labels_path, title, suffix))
            break
