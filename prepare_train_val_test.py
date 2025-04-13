import csv
import os
import random


def prepare_train_val_test_csv():
    filenames = os.listdir("/home/gleb/learning/Segmentation/dataset/pictures")
    random.shuffle(filenames)
    with open("./segmentation_train_set.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for index in range(int(0.85 * len(filenames))):
            writer.writerow([filenames[index]])

    with open("./segmentation_validation_set.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for index in range(int(0.85 * len(filenames)), len(filenames)):
            writer.writerow([filenames[index]])


prepare_train_val_test_csv()
