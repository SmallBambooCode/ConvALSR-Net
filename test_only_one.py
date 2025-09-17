import torch
import argparse
import numpy as np
import cv2
import albumentations as albu
from albumentations.pytorch import ToTensorV2
from pathlib import Path
from train_supervision import Supervision_Train
from tools.cfg import py2cfg
import os

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image_path", type=Path, required=True, help="Path to input image")
    parser.add_argument("-c", "--config_path", type=Path, required=True, help="Path to model config")
    parser.add_argument("-o", "--output_path", type=Path, required=True, help="Path to save predicted mask")
    parser.add_argument("-t", "--tta", default=None, choices=[None, "d4", "lr"], help="Test time augmentation")
    parser.add_argument("-d", "--dataset", default="pv", choices=["pv", "landcoverai", "uavid", "building"], help="Dataset type")
    return parser.parse_args()


def building_to_rgb(mask):
    h, w = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    mask_convert = mask[np.newaxis, :, :]
    mask_rgb[np.all(mask_convert == 0, axis=0)] = [255, 255, 255]
    mask_rgb[np.all(mask_convert == 1, axis=0)] = [0, 0, 0]
    return mask_rgb


def pv2rgb(mask):  # Potsdam and vaihingen
    h, w = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    mask_convert = mask[np.newaxis, :, :]
    mask_rgb[np.all(mask_convert == 3, axis=0)] = [0, 255, 0]
    mask_rgb[np.all(mask_convert == 0, axis=0)] = [255, 255, 255]
    mask_rgb[np.all(mask_convert == 1, axis=0)] = [255, 0, 0]
    mask_rgb[np.all(mask_convert == 2, axis=0)] = [255, 255, 0]
    mask_rgb[np.all(mask_convert == 4, axis=0)] = [0, 204, 255]
    mask_rgb[np.all(mask_convert == 5, axis=0)] = [0, 0, 255]
    return mask_rgb


def landcoverai_to_rgb(mask):
    w, h = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(w, h, 3), dtype=np.uint8)
    mask_convert = mask[np.newaxis, :, :]
    mask_rgb[np.all(mask_convert == 3, axis=0)] = [255, 255, 255]
    mask_rgb[np.all(mask_convert == 0, axis=0)] = [233, 193, 133]
    mask_rgb[np.all(mask_convert == 1, axis=0)] = [255, 0, 0]
    mask_rgb[np.all(mask_convert == 2, axis=0)] = [0, 255, 0]
    mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)
    return mask_rgb


def uavid2rgb(mask):
    h, w = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    mask_convert = mask[np.newaxis, :, :]
    mask_rgb[np.all(mask_convert == 0, axis=0)] = [128, 0, 0]
    mask_rgb[np.all(mask_convert == 1, axis=0)] = [128, 64, 128]
    mask_rgb[np.all(mask_convert == 2, axis=0)] = [0, 128, 0]
    mask_rgb[np.all(mask_convert == 3, axis=0)] = [128, 128, 0]
    mask_rgb[np.all(mask_convert == 4, axis=0)] = [64, 0, 128]
    mask_rgb[np.all(mask_convert == 5, axis=0)] = [192, 0, 192]
    mask_rgb[np.all(mask_convert == 6, axis=0)] = [64, 64, 0]
    mask_rgb[np.all(mask_convert == 7, axis=0)] = [0, 0, 0]
    mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)
    return mask_rgb

def preprocess_image(image_path):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = albu.Compose([
        albu.Normalize(),
        ToTensorV2()
    ])
    augmented = transform(image=image)
    image_tensor = augmented["image"].unsqueeze(0).float()  # 添加 batch 维度
    return image_tensor, image

def postprocess_mask(mask, dataset):
    mask = mask.squeeze(0).cpu().numpy().argmax(0)  # 取类别索引
    if dataset == "pv":
        return pv2rgb(mask)  # 假设有函数转换为 RGB 格式
    elif dataset == "landcoverai":
        return landcoverai_to_rgb(mask)
    elif dataset == "uavid":
        return uavid2rgb(mask)
    elif dataset == "building":
        return building_to_rgb(mask)
    else:
        return mask.astype(np.uint8) * 50  # 默认灰度输出

def main():
    args = get_args()
    config = py2cfg(args.config_path)
    
    model = Supervision_Train.load_from_checkpoint(
        os.path.join(config.weights_path, config.test_weights_name + ".ckpt"), config=config
    )
    model.cuda().eval()
    
    image_tensor, original_image = preprocess_image(args.image_path)
    image_tensor = image_tensor.cuda()
    
    with torch.no_grad():
        output = model(image_tensor)
        output_mask = torch.nn.functional.softmax(output, dim=1)
    
    predicted_mask = postprocess_mask(output_mask, args.dataset)
    
    output_file = args.output_path / f"{args.image_path.stem}_mask.png"
    cv2.imwrite(str(output_file), cv2.cvtColor(predicted_mask, cv2.COLOR_RGB2BGR))
    print(f"Saved mask to {output_file}")

if __name__ == "__main__":
    main()
