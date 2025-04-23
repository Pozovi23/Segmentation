import cv2
import numpy as np
import torch
from torchvision import transforms


def inference(model, picture_path, save_path, label_path = "", write=False, draw_orig=False):
    img = cv2.imread(picture_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.336, 0.340, 0.297), std=(0.192, 0.183, 0.183)),
        ]
    )
    model.eval()

    crop_tensor = transform(img_rgb).unsqueeze(0).cuda()
    with torch.no_grad():
        crop_output = model(crop_tensor)
        pred_mask = crop_output.squeeze().cpu().numpy()
        pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255

    yellow_mask = np.zeros_like(img)
    yellow_mask[pred_mask == 255] = [0, 255, 255]
    img = cv2.imread(picture_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = cv2.addWeighted(img, 0.7, yellow_mask, 0.5, 0)

    if draw_orig and label_path != "":
        white_mask = np.zeros_like(img)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        border = np.zeros((img.shape[0], 10, 3), dtype=np.uint8)
        white_mask[label == 255] = [255, 255, 255]
        img = cv2.addWeighted(img, 0.7, white_mask, 0.5, 0)
        result = np.hstack([result,border, img])

    if write:
        cv2.imwrite(save_path, result)

    return result, pred_mask
