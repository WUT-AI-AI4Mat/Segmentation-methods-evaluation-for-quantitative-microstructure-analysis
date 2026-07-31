import sys
import os
import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from unittest.mock import MagicMock


current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
workspace_root = os.path.dirname(current_dir)


sys.modules["gala"] = MagicMock()
sys.modules["gala.evaluate"] = MagicMock()

if current_dir not in sys.path: sys.path.append(current_dir)
if workspace_root not in sys.path: sys.path.insert(0, workspace_root)

try:
    from utils.segment_anything_ import sam_model_registry
    from utils.segment_anything_.utils.transforms import ResizeLongestSide
    from peft import LoraConfig, get_peft_model
    from Myutils.visualizer import Visualizer
    from Myutils.metrics import Metric
    print(" 依赖与自定义库加载成功！")
except ImportError as e:
    print(f" 导入失败: {e}")
    sys.exit(1)


IMG_DIR = None
LBL_DIR = None


RESULT_ROOT = None
SAVE_PLOT_DIR = None
EXCEL_PATH = None


MODEL_TYPE = "vit_h"
CHECKPOINT_PATH = None

FINETUNED_WEIGHT_PATH = None

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 5


class SemanticMatSAMWrapper(nn.Module):
    def __init__(self, sam_model, num_classes):
        super().__init__()
        self.sam_model = sam_model
        self.num_classes = num_classes
        self.class_tokens = nn.Embedding(num_classes, 256)

    def forward(self, images):
        B = images.shape[0]
        if B > 1: raise ValueError("仅支持 batch_size=1")

        image_embeddings = self.sam_model.image_encoder(images)

        with torch.no_grad():
            _, dense_embeddings = self.sam_model.prompt_encoder(points=None, boxes=None, masks=None)
            image_pe = self.sam_model.prompt_encoder.get_dense_pe()

        sparse_embeddings = self.class_tokens.weight.unsqueeze(1)

        low_res_masks, _ = self.sam_model.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False
        )

        logits = low_res_masks.view(B, self.num_classes, low_res_masks.shape[-2], low_res_masks.shape[-1])
        return logits


LABEL_RULES = [
    lambda x: x,
    lambda x: x.replace("RG", "RGMask"),
    lambda x: x + "_mask",
    lambda x: x + "_label",
    lambda x: x + "_seg",
]
SUPPORTED_EXTS = ['.png', '.jpg', '.jpeg', '.tif', '.bmp']

def cv_imread(file_path, flags=cv2.IMREAD_COLOR):
    if not os.path.exists(file_path): return None
    try:
        img_array = np.fromfile(file_path, dtype=np.uint8)
        img = cv2.imdecode(img_array, flags)
    except: pass
    if img is None: return None
    if flags == cv2.IMREAD_UNCHANGED or flags == -1: return img
    if flags == cv2.IMREAD_GRAYSCALE and img.ndim == 3: img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif flags == cv2.IMREAD_COLOR and img.ndim == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img

def find_label_file(img_name, label_dir):
    base_name = os.path.splitext(img_name)[0]
    for rule_func in LABEL_RULES:
        try: target_name = rule_func(base_name)
        except: continue
        for ext in SUPPORTED_EXTS:
            test_path = os.path.join(label_dir, target_name + ext)
            if os.path.exists(test_path): return test_path
    return None


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate class-token fine-tuned MatSAM.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--finetuned-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--device", default=DEVICE)
    return parser.parse_args()


def main():
    global IMG_DIR, LBL_DIR, RESULT_ROOT, SAVE_PLOT_DIR, EXCEL_PATH
    global CHECKPOINT_PATH, FINETUNED_WEIGHT_PATH, NUM_CLASSES, DEVICE

    args = parse_args()
    IMG_DIR = os.path.join(args.dataset_root, "test", "images")
    LBL_DIR = os.path.join(args.dataset_root, "test", "masks")
    RESULT_ROOT = args.output_dir
    SAVE_PLOT_DIR = os.path.join(RESULT_ROOT, "plots")
    EXCEL_PATH = os.path.join(RESULT_ROOT, "metrics_summary.xlsx")
    CHECKPOINT_PATH = args.checkpoint
    FINETUNED_WEIGHT_PATH = args.finetuned_checkpoint
    NUM_CLASSES = args.num_classes
    DEVICE = args.device

    print("Starting fine-tuned MatSAM evaluation.")
    os.makedirs(SAVE_PLOT_DIR, exist_ok=True)


    print("Building the model and loading the fine-tuned checkpoint.")
    base_sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)

    lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["qkv", "proj"], lora_dropout=0.05, bias="none")
    base_sam.image_encoder = get_peft_model(base_sam.image_encoder, lora_config)

    model = SemanticMatSAMWrapper(base_sam, num_classes=NUM_CLASSES).to(DEVICE)


    if os.path.exists(FINETUNED_WEIGHT_PATH):
        checkpoint = torch.load(FINETUNED_WEIGHT_PATH, map_location=DEVICE)

        model.load_state_dict(checkpoint, strict=False)
        print("Loaded LoRA, decoder, and class-token parameters.")
    else:
        print(f" 找不到权重: {FINETUNED_WEIGHT_PATH}")
        return

    model.eval()


    pixel_mean = base_sam.pixel_mean.to(DEVICE)
    pixel_std = base_sam.pixel_std.to(DEVICE)
    transform = ResizeLongestSide(1024)


    valid_exts = ('.jpg', '.jpeg', '.png', '.tif', '.bmp')
    img_files = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(valid_exts)]
    print(f" 找到 {len(img_files)} 张测试图片")

    all_metrics = []
    start_time = time.time()


    with torch.no_grad():
        for img_file in tqdm(img_files, desc="Processing"):
            img_path = os.path.join(IMG_DIR, img_file)

            try:

                image = cv_imread(img_path, cv2.IMREAD_COLOR)
                if image is None: continue
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                label_path = find_label_file(img_file, LBL_DIR)
                gt_label = None
                if label_path:
                    gt_label = cv_imread(label_path, cv2.IMREAD_UNCHANGED)
                    if gt_label is not None and gt_label.ndim == 3:
                        gt_label = cv2.cvtColor(gt_label, cv2.COLOR_BGR2GRAY)

                if gt_label is not None:
                    h_img, w_img = image_rgb.shape[:2]
                    h_lbl, w_lbl = gt_label.shape[:2]
                    common_h, common_w = min(h_img, h_lbl), min(w_img, w_lbl)
                    image_rgb = image_rgb[:common_h, :common_w]
                    gt_label = gt_label[:common_h, :common_w]

                h_raw, w_raw = image_rgb.shape[:2]


                input_image = transform.apply_image(image_rgb)
                input_tensor = torch.as_tensor(input_image).permute(2, 0, 1).contiguous()


                valid_h, valid_w = input_tensor.shape[1], input_tensor.shape[2]


                padh = 1024 - valid_h
                padw = 1024 - valid_w


                input_tensor = input_tensor.unsqueeze(0).to(DEVICE).float()
                input_tensor = (input_tensor - pixel_mean) / pixel_std


                input_tensor = F.pad(input_tensor, (0, padw, 0, padh))


                logits = model(input_tensor)



                logits_1024 = F.interpolate(logits, size=(1024, 1024), mode='bilinear', align_corners=False)

                logits_valid = logits_1024[:, :, :valid_h, :valid_w]

                logits_upsampled = F.interpolate(logits_valid, size=(h_raw, w_raw), mode='bilinear', align_corners=False)


                pred_mask = logits_upsampled.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)


                if gt_label is not None:
                    current_metric = Metric.compute_all(gt_label, pred_mask, num_classes=NUM_CLASSES)
                    current_metric['filename'] = img_file
                    all_metrics.append(current_metric)

                base_name = os.path.splitext(img_file)[0]
                raw_save_path = os.path.join(SAVE_PLOT_DIR, f"{base_name}_raw_mask.png")


                Visualizer.save_raw_prediction(
                    image_shape=image_rgb.shape,
                    pred_result=pred_mask,
                    save_path=raw_save_path
                )

                plt.close('all')

            except Exception as e:
                print(f"\n 处理出错 {img_file}: {e}")
                import traceback
                traceback.print_exc()
                continue


    total_time = time.time() - start_time
    print(f"\nInference completed in {total_time:.2f} seconds.")

    if len(all_metrics) > 0:
        df = pd.DataFrame(all_metrics)
        cols = ['filename'] + [c for c in df.columns if c != 'filename']
        df = df[cols]
        mean_row = df.select_dtypes(include=[np.number]).mean()
        mean_row['filename'] = 'AVERAGE'

        df_final = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)
        df_final.to_excel(EXCEL_PATH, index=False)
        print(f" Excel 已保存至: {EXCEL_PATH}")
        print(mean_row[['miou', 'mbss', 'dice', 'hd95', 'nsd']])

if __name__ == "__main__":
    main()
