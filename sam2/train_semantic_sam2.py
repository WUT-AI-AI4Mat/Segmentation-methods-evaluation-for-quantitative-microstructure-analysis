import os
import argparse
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

try:
    from sam2.build_sam import build_sam2
    from peft import LoraConfig, get_peft_model

    from segment_anything.utils.transforms import ResizeLongestSide
    print("Loaded SAM2, PEFT, and image transforms.")
except ImportError as e:
    print(f"Failed to import a required dependency: {e}")
    sys.exit(1)


current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


TRAIN_IMG_DIR = None
TRAIN_LBL_DIR = None
VAL_IMG_DIR = None
VAL_LBL_DIR = None


SAVE_DIR = None


CHECKPOINT_PATH = None
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_b+.yaml"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 3


NUM_EPOCHS = 200
LEARNING_RATE = 1e-4
MIN_LR = 1e-6
WEIGHT_DECAY = 1e-4
PATIENCE = 50


class SemanticSAM2Dataset(Dataset):
    def __init__(self, img_dir, lbl_dir, is_train=True):
        self.img_dir = img_dir
        self.lbl_dir = lbl_dir
        self.is_train = is_train
        valid_exts = ('.jpg', '.jpeg', '.png', '.tif', '.bmp')
        self.img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(valid_exts)]
        self.target_size = 1024
        self.transform = ResizeLongestSide(self.target_size)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
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

        image = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = cv2.imread(lbl_path, cv2.IMREAD_UNCHANGED)
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
        input_size = input_image_torch.shape[1:]

        padh = self.target_size - input_image_torch.shape[1]
        padw = self.target_size - input_image_torch.shape[2]
        input_image_torch = F.pad(input_image_torch, (0, padw, 0, padh))

        return {
            "image": input_image_torch.float(),
            "label": torch.tensor(label, dtype=torch.long),
            "original_size": original_size,
            "input_size": input_size,
        }

def custom_collate(batch):
    return batch


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

def calc_multiclass_loss(pred_logits, true_labels):
    ce_loss = F.cross_entropy(pred_logits, true_labels)


    pred_probs = F.softmax(pred_logits, dim=1)
    true_one_hot = F.one_hot(true_labels, num_classes=NUM_CLASSES).permute(0, 3, 1, 2).float()

    intersection = (pred_probs * true_one_hot).sum(dim=(2, 3))
    union = pred_probs.sum(dim=(2, 3)) + true_one_hot.sum(dim=(2, 3))
    dice = (2. * intersection + 1e-5) / (union + 1e-5)
    dice_loss = 1.0 - dice.mean()

    return ce_loss + dice_loss


def forward_pass(data, wrapper_model, is_train=True):
    images = data["image"].unsqueeze(0).to(DEVICE, non_blocking=True)
    gt_labels = data["label"].unsqueeze(0).to(DEVICE, non_blocking=True)
    original_size = data["original_size"]
    input_h, input_w = data["input_size"]


    pixel_mean = torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1).to(DEVICE)
    pixel_std = torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1).to(DEVICE)
    images = (images - pixel_mean) / pixel_std
    with torch.set_grad_enabled(is_train):

        logits = wrapper_model(images)
        logits_1024 = F.interpolate(
            logits, size=(1024, 1024), mode="bilinear", align_corners=False
        )
        logits_valid = logits_1024[:, :, :input_h, :input_w]

        h, w = int(original_size[0]), int(original_size[1])
        logits_upsampled = F.interpolate(
            logits_valid, size=(h, w), mode="bilinear", align_corners=False
        )


        loss = calc_multiclass_loss(logits_upsampled, gt_labels)

    return loss


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune SAM2 with semantic class tokens.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default=MODEL_CFG)
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--device", default=DEVICE)
    return parser.parse_args()


def main():
    global TRAIN_IMG_DIR, TRAIN_LBL_DIR, VAL_IMG_DIR, VAL_LBL_DIR
    global SAVE_DIR, CHECKPOINT_PATH, MODEL_CFG, NUM_CLASSES, DEVICE

    args = parse_args()
    TRAIN_IMG_DIR = os.path.join(args.dataset_root, "train", "images")
    TRAIN_LBL_DIR = os.path.join(args.dataset_root, "train", "masks")
    VAL_IMG_DIR = os.path.join(args.dataset_root, "val", "images")
    VAL_LBL_DIR = os.path.join(args.dataset_root, "val", "masks")
    SAVE_DIR = args.output_dir
    CHECKPOINT_PATH = args.checkpoint
    MODEL_CFG = args.config
    NUM_CLASSES = args.num_classes
    DEVICE = args.device
    os.makedirs(SAVE_DIR, exist_ok=True)

    print("Starting SAM2 semantic class-token fine-tuning.")
    is_terminal = sys.stdout.isatty()

    train_dataset = SemanticSAM2Dataset(TRAIN_IMG_DIR, TRAIN_LBL_DIR, is_train=True)
    val_dataset = SemanticSAM2Dataset(VAL_IMG_DIR, VAL_LBL_DIR, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=custom_collate, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate, num_workers=2)


    base_sam = build_sam2(MODEL_CFG, CHECKPOINT_PATH, device=DEVICE)
    for param in base_sam.parameters(): param.requires_grad = False


    lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["qkv", "proj"], lora_dropout=0.05, bias="none")
    base_sam.image_encoder = get_peft_model(base_sam.image_encoder, lora_config)


    model = SemanticSAM2Wrapper(base_sam, num_classes=NUM_CLASSES).to(DEVICE)


    for param in model.sam_model.sam_prompt_encoder.parameters(): param.requires_grad = False
    for param in model.sam_model.sam_mask_decoder.parameters(): param.requires_grad = True
    print("Training LoRA adapters, the mask decoder, and class tokens.")


    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = AdamW(trainable_params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=MIN_LR)

    best_val_loss = float('inf')
    early_stop_counter = 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0

        current_lr = optimizer.param_groups[0]['lr']
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{NUM_EPOCHS}] (LR: {current_lr:.2e})", disable=not is_terminal)

        for step, batch in enumerate(pbar):
            data = batch[0]
            loss = forward_pass(data, model, is_train=True)

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
                v_loss = forward_pass(data, model, is_train=False)
                val_loss += v_loss.item()

        avg_val_loss = val_loss / len(val_loader)

        status_msg = f" Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}"

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
            status_msg += "  [NEW BEST]"


            save_path = os.path.join(SAVE_DIR, "semantic_sam2_best.pth")
            trainable_state_dict = {k: v.cpu() for k, v in model.named_parameters() if v.requires_grad}
            torch.save(trainable_state_dict, save_path)
        else:
            early_stop_counter += 1
            status_msg += f"  [EarlyStop: {early_stop_counter}/{PATIENCE}]"

        print(status_msg)

        if early_stop_counter >= PATIENCE:
            print("\nEarly stopping triggered.")
            break

if __name__ == "__main__":
    main()
