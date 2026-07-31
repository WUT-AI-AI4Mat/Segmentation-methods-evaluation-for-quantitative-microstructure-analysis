import sys
import os
import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import tifffile
from PIL import Image


current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(current_dir)
if project_root not in sys.path: sys.path.insert(0, project_root)

micro_sam_repo_path = os.path.join(current_dir, "micro_sam")
if os.path.exists(micro_sam_repo_path) and micro_sam_repo_path not in sys.path:
    sys.path.insert(0, micro_sam_repo_path)

try:
    from micro_sam.util import get_sam_model
    from peft import LoraConfig, get_peft_model
    from segment_anything.utils.transforms import ResizeLongestSide
    print(" 成功导入: MicroSAM, PEFT (LoRA) & Transforms")
except ImportError as e:
    print(f" 导入失败: {e}")
    sys.exit(1)


TRAIN_IMG_DIR = None
TRAIN_LBL_DIR = None
VAL_IMG_DIR = None
VAL_LBL_DIR = None

SAVE_DIR = None

MODEL_TYPE = "vit_b_lm"
CHECKPOINT_PATH = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 3

EPOCHS = 200
LR = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 50


def cv_imread(file_path, flags=cv2.IMREAD_COLOR):
    if not os.path.exists(file_path): return None
    img = None
    try:
        img_array = np.fromfile(file_path, dtype=np.uint8)
        img = cv2.imdecode(img_array, flags)
    except: pass

    if img is None and file_path.lower().endswith(('.tif', '.tiff')):
        try:
            img = tifffile.imread(file_path)
            if flags == cv2.IMREAD_COLOR and img.ndim == 3:
                 img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        except: pass

    if img is None:
        try:
            pil_img = Image.open(file_path)
            img = np.asarray(pil_img)
            if flags == cv2.IMREAD_COLOR and img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        except: pass

    if img is None: return None
    if flags == cv2.IMREAD_UNCHANGED or flags == -1: return img

    if flags == cv2.IMREAD_GRAYSCALE and img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif flags == cv2.IMREAD_COLOR and img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img

class SemanticMicroSAMDataset(Dataset):
    def __init__(self, img_dir, lbl_dir, target_size=1024, is_train=True):
        self.img_dir = img_dir
        self.lbl_dir = lbl_dir
        self.is_train = is_train
        valid_exts = ('.jpg', '.png', '.tif', '.bmp')
        self.img_names = [f for f in os.listdir(img_dir) if f.lower().endswith(valid_exts)]
        self.target_size = target_size
        self.transform = ResizeLongestSide(target_size)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)

        base_name = os.path.splitext(img_name)[0]
        lbl_path = None
        for suffix in ["", "_mask", "Mask", "_label", "_seg"]:
            for ext in ['.png', '.tif', '.bmp']:
                tmp = os.path.join(self.lbl_dir, base_name + suffix + ext)
                if os.path.exists(tmp):
                    lbl_path = tmp
                    break
            if lbl_path: break

        image = cv_imread(img_path, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = cv_imread(lbl_path, cv2.IMREAD_UNCHANGED)
        if label.ndim == 3: label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)

        if self.is_train:
            if np.random.rand() > 0.5:
                image_rgb = cv2.flip(image_rgb, 1)
                label = cv2.flip(label, 1)
            if np.random.rand() > 0.5:
                image_rgb = cv2.flip(image_rgb, 0)
                label = cv2.flip(label, 0)
            if np.random.rand() > 0.5:
                image_rgb = cv2.rotate(image_rgb, cv2.ROTATE_90_CLOCKWISE)
                label = cv2.rotate(label, cv2.ROTATE_90_CLOCKWISE)

        original_size = image_rgb.shape[:2]

        input_image = self.transform.apply_image(image_rgb)


        input_image_torch = torch.as_tensor(input_image).permute(2, 0, 1).contiguous()

        return {
            "image": input_image_torch.float(),
            "label": torch.tensor(label, dtype=torch.long),
            "original_size": original_size
        }

def custom_collate(batch):
    return batch


class SemanticMicroSAMWrapper(nn.Module):
    def __init__(self, sam_model, num_classes):
        super().__init__()
        self.sam_model = sam_model
        self.num_classes = num_classes
        self.class_tokens = nn.Embedding(num_classes, 256)

    def forward(self, images):
        B = images.shape[0]
        if B > 1:
            raise ValueError("由于 SAM Decoder 的底层并发机制限制，当前仅支持 batch_size=1。")

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

def calc_multiclass_loss(pred_logits, true_labels):
    ce_loss = F.cross_entropy(pred_logits, true_labels)

    pred_probs = F.softmax(pred_logits, dim=1)
    true_one_hot = F.one_hot(true_labels, num_classes=NUM_CLASSES).permute(0, 3, 1, 2).float()

    intersection = (pred_probs * true_one_hot).sum(dim=(2, 3))
    union = pred_probs.sum(dim=(2, 3)) + true_one_hot.sum(dim=(2, 3))


    dice = (2. * intersection + 1.0) / (union + 1.0)
    dice_loss = 1.0 - dice.mean()

    return ce_loss + dice_loss


def forward_pass(data, wrapper_model, sam_base, is_train=True):
    images = data["image"].unsqueeze(0).to(DEVICE, non_blocking=True)
    gt_labels = data["label"].unsqueeze(0).to(DEVICE, non_blocking=True)
    original_size = data["original_size"]


    pixel_mean = sam_base.pixel_mean.to(DEVICE)
    pixel_std = sam_base.pixel_std.to(DEVICE)
    images = (images - pixel_mean) / pixel_std


    input_h, input_w = images.shape[2], images.shape[3]


    padh = 1024 - input_h
    padw = 1024 - input_w
    images = F.pad(images, (0, padw, 0, padh))

    with torch.set_grad_enabled(is_train):
        logits = wrapper_model(images)


        logits_1024 = F.interpolate(logits, size=(1024, 1024), mode="bilinear", align_corners=False)
        logits_valid = logits_1024[:, :, :input_h, :input_w]

        h, w = int(original_size[0]), int(original_size[1])
        logits_upsampled = F.interpolate(logits_valid, size=(h, w), mode="bilinear", align_corners=False)

        loss = calc_multiclass_loss(logits_upsampled, gt_labels)

    return loss


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune micro-sam with semantic class tokens.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--model-type", default=MODEL_TYPE)
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--device", default=DEVICE)
    return parser.parse_args()


def main():
    global TRAIN_IMG_DIR, TRAIN_LBL_DIR, VAL_IMG_DIR, VAL_LBL_DIR
    global SAVE_DIR, CHECKPOINT_PATH, MODEL_TYPE, NUM_CLASSES, DEVICE

    args = parse_args()
    TRAIN_IMG_DIR = os.path.join(args.dataset_root, "train", "images")
    TRAIN_LBL_DIR = os.path.join(args.dataset_root, "train", "masks")
    VAL_IMG_DIR = os.path.join(args.dataset_root, "val", "images")
    VAL_LBL_DIR = os.path.join(args.dataset_root, "val", "masks")
    SAVE_DIR = args.output_dir
    CHECKPOINT_PATH = args.checkpoint
    MODEL_TYPE = args.model_type
    NUM_CLASSES = args.num_classes
    DEVICE = args.device
    os.makedirs(SAVE_DIR, exist_ok=True)

    print("Starting micro-sam semantic class-token fine-tuning.")
    is_terminal = sys.stdout.isatty()

    predictor = get_sam_model(model_type=MODEL_TYPE, checkpoint_path=CHECKPOINT_PATH, device=DEVICE)
    sam_base = predictor.model

    lora_config = LoraConfig(
        r=8, lora_alpha=16, target_modules=["qkv", "proj"],
        lora_dropout=0.05, bias="none"
    )
    sam_base.image_encoder = get_peft_model(sam_base.image_encoder, lora_config)

    model = SemanticMicroSAMWrapper(sam_base, num_classes=NUM_CLASSES).to(DEVICE)

    for param in model.sam_model.prompt_encoder.parameters(): param.requires_grad = False
    for param in model.sam_model.mask_decoder.parameters(): param.requires_grad = True
    print("Trainable modules: image encoder LoRA, mask decoder, and class tokens.")

    train_dataset = SemanticMicroSAMDataset(TRAIN_IMG_DIR, TRAIN_LBL_DIR, is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=custom_collate, num_workers=4, pin_memory=True)

    val_dataset = SemanticMicroSAMDataset(VAL_IMG_DIR, VAL_LBL_DIR, is_train=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate, num_workers=2, pin_memory=True)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    best_val_loss = float('inf')
    no_improve_count = 0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0

        current_lr = optimizer.param_groups[0]['lr']
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train] (LR: {current_lr:.2e})", disable=not is_terminal)

        for batch in pbar:
            data = batch[0]
            loss = forward_pass(data, model, sam_base, is_train=True)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            if is_terminal:
                pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        avg_train_loss = train_loss / len(train_loader)
        scheduler.step()

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                data = batch[0]
                v_loss = forward_pass(data, model, sam_base, is_train=False)
                val_loss += v_loss.item()

        avg_val_loss = val_loss / len(val_loader)

        status_msg = f" Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}"

        if avg_val_loss < best_val_loss:
            no_improve_count = 0
            best_val_loss = avg_val_loss
            status_msg += "  [NEW BEST]"

            save_path = os.path.join(SAVE_DIR, "semantic_microsam_best.pth")
            trainable_state_dict = {k: v.cpu() for k, v in model.named_parameters() if v.requires_grad}
            torch.save(trainable_state_dict, save_path)
        else:
            no_improve_count += 1
            status_msg += f"  [EarlyStop: {no_improve_count}/{PATIENCE}]"

        print(status_msg)

        if no_improve_count >= PATIENCE:
            print(f"\n 触发早停机制！验证集 Loss 已经连续 {PATIENCE} 轮没有下降。")
            break

if __name__ == "__main__":
    main()
