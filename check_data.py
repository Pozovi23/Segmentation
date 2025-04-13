import base64
import hashlib
import io
import json
import os

import cv2
import numpy as np
import requests
from PIL import Image


def hash_image(image_path):
    with open(image_path, "rb") as image_file:
        hasher = hashlib.md5()
        image = image_file.read()
        hasher.update(image)
        digest = hasher.digest()

    return digest


def find_duplicates(main_path):
    duplicates = []
    for folder in os.listdir(main_path):
        hashes_and_their_photos = {}
        for file in os.listdir(main_path + "/" + folder):
            current_path = main_path + "/" + folder + "/" + file
            hashed_img = hash_image(current_path)
            if hashes_and_their_photos.get(hashed_img) is None:
                hashes_and_their_photos[hashed_img] = [current_path]
            else:
                print(
                    hashes_and_their_photos[hashed_img][0], "          ", current_path
                )
                duplicates.append(current_path)

    return duplicates


def check_file_extension(main_path):
    files_with_wrong_extension = []
    for folder in os.listdir(main_path):
        for file in os.listdir(main_path + "/" + folder):
            all_path = main_path + "/" + folder + "/" + file
            if not all_path.endswith((".tiff", ".tif")):
                files_with_wrong_extension.append(all_path)

    return files_with_wrong_extension


def visualise_masks():
    dataset_path = "/home/gleb/Загрузки/archive/tiff"
    pictures_dir = os.path.join(dataset_path, "train")
    labels_dir = os.path.join(dataset_path, "train_labels")
    output_dir = os.path.join("picture_and_mask_new")

    for image_file in os.listdir(pictures_dir):
        image_path = os.path.join(pictures_dir, image_file)
        mask_path = os.path.join(labels_dir, image_file)

        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path[:-1], cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(image_path)
        elif mask is None:
            print(image_path)
        else:
            overlay = image.copy()
            overlay[mask > 0] = [0, 255, 255]

            alpha = 0.3
            combined = np.hstack(
                (cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0), image)
            )
            output_path = os.path.join(output_dir, image_file)
            cv2.imwrite(output_path, combined)


def tiff_to_png():
    pictures_dir = "/home/gleb/learning/Segmentation/segmentations/"
    output_dir = "/home/gleb/learning/Segmentation/segmentations/"
    i = 0
    for file in os.listdir(pictures_dir):
        image = cv2.imread(pictures_dir + file)
        image = cv2.resize(image, (400, 400))
        cv2.imwrite(output_dir + f"{i}.png", image)
        i += 1
        if i == 4:
            break
tiff_to_png()

def delete_old_masks():
    path = "/home/gleb/learning/Segmentation/exported_masks/done_masks"
    for filename in os.listdir(path):
        os.remove("/home/gleb/learning/Segmentation/dataset/labels_png/" + filename)


def divide_into_two_parts():
    pictures_dir = "/home/gleb/learning/Segmentation/dataset/labels_png"

    list_file = "delete.txt"

    with open(list_file, "r") as f:
        filenames = [line.strip() for line in f if line.strip()]

    for name in filenames:
        file_path = os.path.join(pictures_dir, name)
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Удалён: {file_path}")
        else:
            print(f"Не найден: {file_path}")

    # output_dir = "/home/gleb/learning/Segmentation/pictures_my_half"
    # files = sorted(os.listdir(pictures_dir))
    # files = files[0:int(len(files) / 2)]
    #
    # for image_file in files:
    #     image_path = os.path.join(pictures_dir, image_file)
    #
    #     image = cv2.imread(image_path)
    #
    #     output_path = os.path.join(output_dir, image_file)
    #     cv2.imwrite(output_path, image)
    #
    # output_dir = "/home/gleb/learning/Segmentation/pictures_sveta_half"
    # files = sorted(os.listdir(pictures_dir))
    # files = files[int(len(files) / 2) : len(files)]
    #
    # for image_file in files:
    #     image_path = os.path.join(pictures_dir, image_file)
    #
    #     image = cv2.imread(image_path)
    #
    #     output_path = os.path.join(output_dir, image_file)
    #     cv2.imwrite(output_path, image)


def labelstudio():
    LABEL_STUDIO_URL = "http://localhost:8080"
    API_TOKEN = "f964fbfaa516e5bcf8e33fc8371aae84f339e2b0"
    PROJECT_ID = 6
    FOLDER_PATH = "/home/gleb/learning/Segmentation/dataset/labels_png"

    headers = {"Authorization": f"Token {API_TOKEN}"}

    for filename in os.listdir(FOLDER_PATH):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        file_path = os.path.join(FOLDER_PATH, filename)

        try:
            with Image.open(file_path) as img:
                img.verify()

            with open(file_path, "rb") as f:
                data = {
                    "data": {
                        "image": f"/data/upload/{filename}",
                        "file_name": filename,
                    }
                }

                files = {
                    "file": (filename, f),
                    "json": (None, json.dumps(data), "application/json"),
                }

                response = requests.post(
                    f"{LABEL_STUDIO_URL}/api/projects/{PROJECT_ID}/import",
                    headers=headers,
                    files=files,
                    timeout=30,
                )

            if response.status_code == 201:
                print(f"{filename}: OK")
            else:
                print(f"{filename}: Error {response.status_code} - {response.text}")

        except Exception as e:
            print(f"{filename}: Critical error - {str(e)}")


def download_labelstudio_masks():
    LABEL_STUDIO_URL = "http://localhost:8080"
    API_TOKEN = "f964fbfaa516e5bcf8e33fc8371aae84f339e2b0"
    PROJECT_ID = 3
    OUTPUT_FOLDER = "/home/gleb/learning/Segmentation/exported_masks"

    # Создаем папку для масок
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    print(f"Output folder: {OUTPUT_FOLDER}")

    headers = {"Authorization": f"Token {API_TOKEN}"}

    try:
        health_check = requests.get(f"{LABEL_STUDIO_URL}/api/health", timeout=5)
        print(f"Server health status: {health_check.status_code}")
    except Exception as e:
        print(f"Cannot connect to Label Studio: {str(e)}")
        return

    project_url = f"{LABEL_STUDIO_URL}/api/projects/{PROJECT_ID}"
    try:
        project_resp = requests.get(project_url, headers=headers, timeout=10)
        if project_resp.status_code != 200:
            print(
                f"Project access error: {project_resp.status_code} - {project_resp.text}"
            )
            return
        print(f"Project found: {project_resp.json().get('title')}")
    except Exception as e:
        print(f"Project check failed: {str(e)}")
        return

    tasks_url = f"{LABEL_STUDIO_URL}/api/projects/{PROJECT_ID}/tasks"
    try:
        print(f"Fetching tasks from: {tasks_url}")
        response = requests.get(
            tasks_url, headers=headers, params={"page_size": 100}, timeout=30
        )

        if response.status_code != 200:
            print(
                f"Tasks fetch error: {response.status_code}\nResponse: {response.text}"
            )
            return

        tasks = response.json()
        print(f"Found {len(tasks)} tasks")

        if not tasks:
            print("No tasks found in the project")
            return

        with open(os.path.join(OUTPUT_FOLDER, "raw_tasks.json"), "w") as f:
            json.dump(tasks, f, indent=2)
        print("Saved raw tasks data to raw_tasks.json")

        saved_count = 0
        for i, task in enumerate(tasks, 1):
            print(f"\nProcessing task {i}/{len(tasks)}")

            original_filename = task.get("data", {}).get("file_name")
            if not original_filename:
                original_filename = task.get("data", {}).get("image", "").split("/")[-1]
                if not original_filename:
                    print(" - Cannot determine filename, skipping")
                    continue

            print(f" - File: {original_filename}")

            annotations = task.get("annotations", [])
            if not annotations:
                print(" - No annotations found")
                continue

            print(f" - Found {len(annotations)} annotations")

            for ann in annotations:
                if not ann.get("result"):
                    continue

                for result in ann["result"]:
                    if result.get("type") == "brushlabels" and "image" in result.get(
                        "value", {}
                    ):
                        try:
                            mask_b64 = result["value"]["image"]
                            mask_data = (
                                mask_b64.split(",")[1] if "," in mask_b64 else mask_b64
                            )

                            mask_image = Image.open(
                                io.BytesIO(base64.b64decode(mask_data))
                            )
                            mask_filename = (
                                os.path.splitext(original_filename)[0] + "_mask.png"
                            )
                            mask_path = os.path.join(OUTPUT_FOLDER, mask_filename)

                            mask_image.save(mask_path)
                            saved_count += 1
                            print(f" - Saved mask: {mask_filename}")
                            break

                        except Exception as e:
                            print(f" - Mask processing error: {str(e)}")
                            continue

        print(f"\nTotal saved masks: {saved_count}/{len(tasks)}")

    except Exception as e:
        print(f"Critical error: {str(e)}")


def rename():
    with open("exported_masks/raw_tasks.json", "r") as f:
        tasks = json.load(f)

    task_id_to_filename = {}
    for task in tasks:
        if "data" in task and "image" in task["data"]:
            filename = task["data"]["image"].split("/")[-1]
            task_id_to_filename[task["id"]] = filename
    print(task_id_to_filename)
    for filename in os.listdir("exported_masks/merged_masks"):
        if filename.startswith("task-") and filename.endswith(".png"):
            task_id = int(filename.split("-")[1].split(".")[0])
            print(task_id)
            if task_id in task_id_to_filename:
                new_name = task_id_to_filename[task_id].split("-")[1]
                print(filename)
                os.rename(
                    "exported_masks/merged_masks/" + filename,
                    "exported_masks/done_masks/" + new_name,
                )
                print(f"Renamed {filename} to {new_name}")


def get_unique_task_ids(mask_dir):
    task_ids = set()
    for filename in os.listdir(mask_dir):
        if filename.startswith("task-") and "-annotation-" in filename:
            try:
                task_id = int(filename.split("-")[1])
                task_ids.add(task_id)
            except (IndexError, ValueError):
                continue
    return sorted(task_ids)


def combine_masks_for_task(task_id, mask_dir, output_dir):
    combined_mask = np.zeros((1500, 1500), dtype=np.uint8)
    mask_files = [
        f
        for f in os.listdir(mask_dir)
        if f.startswith(f"task-{task_id}-annotation")
        and f.endswith((".png", ".jpg", ".tif"))
    ]

    if not mask_files:
        print(f"⚠ Нет масок для task_id = {task_id}")
        return

    for mask_file in mask_files:
        mask_path = os.path.join(mask_dir, mask_file)
        mask = np.array(Image.open(mask_path).convert("L"))
        combined_mask = np.maximum(combined_mask, mask)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"task-{task_id}-combined-mask.png")
    Image.fromarray(combined_mask).save(output_path)
    print(f"Объединённая маска сохранена: {output_path}")


def combine_all_masks(mask_dir, output_dir):
    task_ids = get_unique_task_ids(mask_dir)
    if not task_ids:
        print("Не найдено ни одного task-id в именах файлов!")
        return

    print(f"Найдены task_ids: {task_ids}")
    for task_id in task_ids:
        combine_masks_for_task(task_id, mask_dir, output_dir)


def combine():
    mask_dir = "/home/gleb/learning/Segmentation/exported_masks/raw_masks"
    output_dir = "/home/gleb/learning/Segmentation/exported_masks/merged_masks"
    combine_all_masks(mask_dir, output_dir)
