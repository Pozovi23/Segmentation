import csv
import os
import random

import cv2
import numpy as np


def make_crops(window_size=224, threshold=0.98):
    input_dir_pictures = "/home/gleb/learning/Segmentation/dataset/src/pictures_png"
    input_dir_labels = "/home/gleb/learning/Segmentation/dataset/src/labels_png"

    output_dir_pictures = "/home/gleb/learning/Segmentation/dataset/pictures"
    output_dir_labels = "/home/gleb/learning/Segmentation/dataset/labels"

    files_list = os.listdir(input_dir_pictures)
    random.shuffle(files_list)

    for image_path in files_list[: int(0.85 * len(files_list))]:

        img = cv2.imread(input_dir_pictures + "/" + image_path)
        label = cv2.imread(input_dir_labels + "/" + image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Не удалось загрузить изображение {image_path}")
            return

        if label is None:
            print(f"Не удалось загрузить label {image_path}")
            return

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        white_pixel = np.array([255, 255, 255])
        crop_count = 0

        image_path = image_path[:-4]
        for y in range(0, img.shape[0] - window_size + 1, window_size):
            for x in range(0, img.shape[1] - window_size + 1, window_size):
                crop = img_rgb[y : y + window_size, x : x + window_size]

                white_pixels = np.sum(np.all(crop == white_pixel, axis=2))
                total_pixels = window_size * window_size
                non_white_ratio = 1 - (white_pixels / total_pixels)

                if non_white_ratio > threshold:
                    crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
                    crop_label = label[y : y + window_size, x : x + window_size]
                    cv2.imwrite(
                        output_dir_pictures
                        + "/"
                        + image_path
                        + f"_crop_{crop_count}.png",
                        crop_bgr,
                    )
                    cv2.imwrite(
                        output_dir_labels
                        + "/"
                        + image_path
                        + f"_crop_{crop_count}.png",
                        crop_label,
                    )
                    crop_count += 1

        print(f"Сохранено {crop_count} кропов в {image_path}")

    with open("./segmentation_test_set.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for image_path in files_list[int(0.85 * len(files_list)) : len(files_list)]:
            writer.writerow([image_path])


make_crops()
