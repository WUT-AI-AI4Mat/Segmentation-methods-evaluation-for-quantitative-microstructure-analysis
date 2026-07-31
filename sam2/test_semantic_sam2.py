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


current_file_path = os.path.abspath(__file__)
sam2_dir = os.path.dirname(current_file_path)
workspace_dir = os.path.dirname(sam2_dir)

if workspace_dir not in sys.path:
    sys.path.insert(0, workspace_dir)

try:
    from sam2.build_sam import build_sam2
    from peft import LoraConfig, get_peft_model
    from segment_anything.utils.transforms import ResizeLongestSide
    from Myutils.visualizer import Visualizer
    from Myutils.metrics import Metric
    print("Loaded SAM2 and evaluation utilities.")
except ImportError as e:
    print(f"Failed to import a required dependency: {e}")
    sys.exit(1)


IMG_DIR = None
LBL_DIR = None


RESULT_ROOT = None
SAVE_PLOT_DIR = None
EXCEL_PATH = None


CHECKPOINT_PATH = None
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_b+.yaml"

FINETUNED_WEIGHT_PATH = None

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 5


class SemanticSAM2Wrapper(nn.Module):
    def __init__(self, sam_model, num_classes):
        super().__init__()
        self.sam_model = sam_model
        self.num_classes = num_classes
        self.class_tokens = nn.Embedding(num_classes, 256)

    def forward(self, images):
        B = images.shape[0]

        image_features_dict = self.sam_model.forward_image(images)
        vision_features = image_features_dict["vision_features"]
        backbone_fpn = image_features_dict["backbone_fpn"]

        with torch.no_grad():
            _, dense_embeddings = self.sam_model.sam_prompt_encoder(points=None, boxes=None, masks=None)
            image_pe = self.sam_model.sam_prompt_encoder.get_dense_pe()

        vision_features_repeat = vision_features.repeat_interleave(self.num_classes, dim=0)
        high_res_features_repeat = [feat.repeat_interleave(self.num_classes, dim=0) for feat in backbone_fpn[:2]]

        dense_embeddings_repeat = dense_embeddings.expand(
            B * self.num_classes, -1, vision_features.shape[-2], vision_features.shape[-1]
        )

        sparse_embeddings = self.class_tokens.weight.unsqueeze(1).repeat(B, 1, 1)

        low_res_masks, _, _, _ = self.sam_model.sam_mask_decoder(
            image_embeddings=vision_features_repeat,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings_repeat,
            multimask_output=False,
            repeat_image=False,
            high_res_features=high_res_features_repeat
        )

        logits = low_res_masks.view(B, self.num_classes, low_res_masks.shape[-2], low_res_masks.shape[-1])
        return logits


LABEL_RULES = [
    lambda x: x,
    lambda x: x.replace("RG", "RGMask"),
    lambda x: x + "_mask",
    lambda x: x + "_label",
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
    parser = argparse.ArgumentParser(description="Evaluate class-token fine-tuned SAM2.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--finetuned-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default=MODEL_CFG)
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--device", default=DEVICE)
    return parser.parse_args()


def main():
    global IMG_DIR, LBL_DIR, RESULT_ROOT, SAVE_PLOT_DIR, EXCEL_PATH
    global CHECKPOINT_PATH, FINETUNED_WEIGHT_PATH, MODEL_CFG, NUM_CLASSES, DEVICE

    args = parse_args()
    IMG_DIR = os.path.join(args.dataset_root, "test", "images")
    LBL_DIR = os.path.join(args.dataset_root, "test", "masks")
    RESULT_ROOT = args.output_dir
    SAVE_PLOT_DIR = os.path.join(RESULT_ROOT, "plots")
    EXCEL_PATH = os.path.join(RESULT_ROOT, "metrics_summary.xlsx")
    CHECKPOINT_PATH = args.checkpoint
    FINETUNED_WEIGHT_PATH = args.finetuned_checkpoint
    MODEL_CFG = args.config
    NUM_CLASSES = args.num_classes
    DEVICE = args.device

    print("Starting fine-tuned SAM2 evaluation.")
    os.makedirs(SAVE_PLOT_DIR, exist_ok=True)


    print("Building the model and loading the fine-tuned checkpoint.")
    base_sam = build_sam2(MODEL_CFG, CHECKPOINT_PATH, device=DEVICE, apply_postprocessing=False)

    lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["qkv", "proj"], lora_dropout=0.05, bias="none")
    base_sam.image_encoder = get_peft_model(base_sam.image_encoder, lora_config)

    model = SemanticSAM2Wrapper(base_sam, num_classes=NUM_CLASSES).to(DEVICE)


    if os.path.exists(FINETUNED_WEIGHT_PATH):
        checkpoint = torch.load(FINETUNED_WEIGHT_PATH, map_location=DEVICE)

        model.load_state_dict(checkpoint, strict=False)
        print("Loaded LoRA, decoder, and class-token parameters.")
    else:
        print(f"Fine-tuned checkpoint not found: {FINETUNED_WEIGHT_PATH}")
        return

    model.eval()


    transform = ResizeLongestSide(1024)
    pixel_mean = torch.tensor([123.675, 116.28, 103.53], device=DEVICE).view(1, 3, 1, 1)
    pixel_std = torch.tensor([58.395, 57.12, 57.375], device=DEVICE).view(1, 3, 1, 1)

    valid_exts = ('.jpg', '.jpeg', '.png', '.tif', '.bmp')
    img_files = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(valid_exts)]
    print(f"Found {len(img_files)} test images.")

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
                valid_h, valid_w = input_tensor.shape[1:]
                input_tensor = input_tensor.unsqueeze(0).to(DEVICE).float()
                input_tensor = (input_tensor - pixel_mean) / pixel_std
                input_tensor = F.pad(
                    input_tensor, (0, 1024 - valid_w, 0, 1024 - valid_h)
                )

                logits = model(input_tensor)
                logits_1024 = F.interpolate(
                    logits, size=(1024, 1024), mode="bilinear", align_corners=False
                )
                logits_valid = logits_1024[:, :, :valid_h, :valid_w]
                logits_upsampled = F.interpolate(
                    logits_valid,
                    size=(h_raw, w_raw),
                    mode="bilinear",
                    align_corners=False,
                )
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
                print(f"\nFailed to process {img_file}: {e}")
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
        print(f"Saved metrics to: {EXCEL_PATH}")
        print(mean_row[['miou', 'mbss', 'dice', 'hd95', 'nsd']])

if __name__ == "__main__":
    main()
