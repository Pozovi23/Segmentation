import csv
import math

import cv2
import numpy as np


def mean_and_std():

    amount_of_pictures = 0
    r_sum = 0.0
    g_sum = 0.0
    b_sum = 0.0

    with open("./segmentation_train_set.csv", "r") as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            amount_of_pictures += 1
            image = cv2.imread(
                "/home/gleb/learning/Segmentation/dataset/pictures/" + line[0]
            )
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            r_sum += image[:, :, 0].mean()
            g_sum += image[:, :, 1].mean()
            b_sum += image[:, :, 2].mean()

    mean_r = r_sum / amount_of_pictures
    mean_g = g_sum / amount_of_pictures
    mean_b = b_sum / amount_of_pictures

    std_r = 0.0
    std_g = 0.0
    std_b = 0.0

    with open("./segmentation_train_set.csv", "r") as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            image = cv2.imread(
                "/home/gleb/learning/Segmentation/dataset/pictures/" + line[0]
            )
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            std_r += np.sum((image[:, :, 0] - mean_r) ** 2)
            std_g += np.sum((image[:, :, 1] - mean_g) ** 2)
            std_b += np.sum((image[:, :, 2] - mean_b) ** 2)

    amount_of_pixels = amount_of_pictures * 224 * 224
    std_r = math.sqrt(std_r / amount_of_pixels) / 255
    std_g = math.sqrt(std_g / amount_of_pixels) / 255
    std_b = math.sqrt(std_b / amount_of_pixels) / 255

    return mean_r / 255, mean_g / 255, mean_b / 255, std_r, std_g, std_b


a = mean_and_std()
# (0.3368874871363464, 0.34021313731280634, 0.29795370005121335, 0.1928318562726643, 0.18339078606561932, 0.1834100538738419)
print(a)
