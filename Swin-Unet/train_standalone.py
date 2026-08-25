import os
import argparse
import sys
import gc
import time
import random
import traceback
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import albumentations as albu
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt


from networks.vision_transformer import SwinUnet as ViT_seg
from config import get_config


BASE_SAVE_DIR = None
PRETRAINED_PATH = None


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


BATCH_SIZE = 32
NUM_WORKERS = 8

EPOCHS = 500
BASE_LR = 1e-4
WEIGHT_DECAY = 1e-4
IMG_SIZE = 224

PATIENCE = 50

class Args:
    cfg = "configs/swin_tiny_patch4_window7_224_lite.yaml"
    opts = None
    zip = False
    cache_mode = 'part'
    resume = None
    accumulation_steps = None
    use_checkpoint = False
    amp_opt_level = 'O1'
    tag = None
    eval = False
    throughput = False
    batch_size = BATCH_SIZE
    base_lr = BASE_LR
    output_dir = BASE_SAVE_DIR
    img_size = IMG_SIZE

args = Args()


class UniversalDataset(Dataset):
    def __init__(self, root_dir, num_classes, mode='train', augmentation=None):
        self.images_dir = os.path.join(root_dir, mode, 'images')
        self.masks_dir = os.path.join(root_dir, mode, 'masks')
        self.num_classes = num_classes

        valid_exts = ('.jpg', '.png', '.tif', '.bmp', '.jpeg')
        if not os.path.exists(self.images_dir):
            raise FileNotFoundError(f"Image directory not found: {self.images_dir}")

        self.image_names = sorted([f for f in os.listdir(self.images_dir) if f.lower().endswith(valid_exts)])
        self.mask_files_map = {f.lower(): f for f in os.listdir(self.masks_dir)}

        self.images_fps = []
        self.masks_fps = []

        self.match_rules = [
            lambda x: x,
            lambda x: x.replace("RG", "RGMask", 1),
            lambda x: x + "_mask",
            lambda x: x + "_label",
            lambda x: x + "_seg",
            lambda x: x.replace("image", "mask"),
            lambda x: x.replace("img", "msk"),
        ]

        for img_name in self.image_names:
            img_path = os.path.join(self.images_dir, img_name)
            img_stem = os.path.splitext(img_name)[0]

            found_mask_filename = None
            for rule in self.match_rules:
                try:
                    candidate_stem = rule(img_stem)
                    for ext in valid_exts:
                        candidate_full = (candidate_stem + ext).lower()
                        if candidate_full in self.mask_files_map:
                            found_mask_filename = self.mask_files_map[candidate_full]
                            break
                except: continue
                if found_mask_filename: break

            if found_mask_filename:
                self.images_fps.append(img_path)
                self.masks_fps.append(os.path.join(self.masks_dir, found_mask_filename))

        print(f"   -> [{mode}] Loaded: {len(self.images_fps)} images (Target Classes: {num_classes})", flush=True)
        self.augmentation = augmentation

        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

    def __getitem__(self, i):
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(self.masks_fps[i], -1)

        if mask.ndim == 3:
            mask = np.max(mask, axis=2)


        if self.num_classes == 2:
            mask = np.where(mask > 0, 1.0, 0.0)
        else:
            mask = mask.astype(np.uint8)

        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        image = image.astype(np.float32) / 255.0
        image = (image - self.mean) / self.std
        image = image.transpose(2, 0, 1)

        mask = mask.astype('int64')

        return {
            'image': torch.from_numpy(image).float(),
            'label': torch.from_numpy(mask).long()
        }

    def __len__(self):
        return len(self.images_fps)


def get_augmentation(mode='train'):
    if mode == 'train':
        return albu.Compose([
            albu.Resize(IMG_SIZE, IMG_SIZE),
            albu.HorizontalFlip(p=0.5),
            albu.VerticalFlip(p=0.5),
            albu.RandomRotate90(p=0.5),
            albu.ShiftScaleRotate(scale_limit=0.1, rotate_limit=15, shift_limit=0.1, p=0.5, border_mode=0),
        ])
    else:
        return albu.Compose([
            albu.Resize(IMG_SIZE, IMG_SIZE)
        ])


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = 2.0 * torch.sum(inputs[:, i] * target[:, i]) / (torch.sum(inputs[:, i]) + torch.sum(target[:, i]) + 1e-5)
            loss += dice * weight[i]
        return 1 - loss / self.n_classes

def calculate_mdice(pred_logits, gt_mask, num_classes):
    pred_mask = torch.argmax(torch.softmax(pred_logits, dim=1), dim=1)
    dice_sum = 0.0
    valid_classes = 0

    for c in range(num_classes):
        p_c = (pred_mask == c).float()
        g_c = (gt_mask == c).float()

        intersection = (p_c * g_c).sum()
        union = p_c.sum() + g_c.sum()


        if union > 0:
            dice_sum += (2.0 * intersection) / (union + 1e-5)
            valid_classes += 1

    return (dice_sum / valid_classes).item() if valid_classes > 0 else 0.0


def run_training_for_dataset(dataset_name, dataset_path, num_classes):
    print(f"\n{'='*20} Task: {dataset_name} | Classes: {num_classes} {'='*20}", flush=True)

    current_save_dir = os.path.join(BASE_SAVE_DIR, dataset_name)
    os.makedirs(current_save_dir, exist_ok=True)

    log_file = os.path.join(current_save_dir, 'train_log.txt')
    if not os.path.exists(log_file):
        with open(log_file, 'w') as f:
            f.write(f"Epoch,Train_Loss,Val_Loss,Val_mDice\n")

    config = get_config(args)
    model = ViT_seg(config, img_size=IMG_SIZE, num_classes=num_classes).to(DEVICE)

    if os.path.exists(PRETRAINED_PATH):
        checkpoint = torch.load(PRETRAINED_PATH, map_location=DEVICE)
        pretrained_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        model_dict = model.state_dict()

        new_state_dict = {}
        matched_cnt = 0
        for k, v in pretrained_dict.items():
            target_key = f"swin_unet.{k}" if f"swin_unet.{k}" in model_dict else k
            if target_key in model_dict:
                if model_dict[target_key].shape == v.shape:
                    new_state_dict[target_key] = v
                    matched_cnt += 1

        model_dict.update(new_state_dict)
        model.load_state_dict(model_dict, strict=False)
        print(f" Loaded {matched_cnt} layers from ImageNet weights.", flush=True)
    else:
        print(" Pretrained weights not found.", flush=True)

    try:
        train_ds = UniversalDataset(dataset_path, num_classes, 'train', augmentation=get_augmentation('train'))

        val_path_check = os.path.join(dataset_path, 'val')
        val_mode = 'val' if os.path.exists(val_path_check) and len(os.listdir(val_path_check)) > 0 else 'train'
        val_ds = UniversalDataset(dataset_path, num_classes, val_mode, augmentation=get_augmentation('val'))

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)


        optimizer = optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

        ce_loss = nn.CrossEntropyLoss()
        dice_loss = DiceLoss(num_classes)
        scaler = GradScaler()

        best_mdice = 0.0
        epochs_no_improve = 0

        for epoch in range(EPOCHS):
            model.train()
            train_loss_sum = 0
            total_batches = len(train_loader)

            print(f"\n========== Epoch {epoch+1}/{EPOCHS} [LR: {optimizer.param_groups[0]['lr']:.2e}] ==========", flush=True)

            for batch_idx, batch in enumerate(train_loader):
                img = batch['image'].to(DEVICE)
                label = batch['label'].to(DEVICE)

                optimizer.zero_grad()
                with autocast():
                    outputs = model(img)
                    loss_ce = ce_loss(outputs, label)
                    loss_dice_val = dice_loss(outputs, label, softmax=True)
                    loss = 0.5 * loss_ce + 0.5 * loss_dice_val

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                train_loss_sum += loss.item()

                if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == total_batches:
                    print(f"   [Batch {batch_idx + 1}/{total_batches}] Loss: {loss.item():.4f}", flush=True)

            avg_train_loss = train_loss_sum / len(train_loader)

            model.eval()
            val_loss_sum = 0
            val_mdice_sum = 0

            with torch.no_grad():
                for batch in val_loader:
                    img = batch['image'].to(DEVICE)
                    label = batch['label'].to(DEVICE)

                    outputs = model(img)

                    loss_ce = ce_loss(outputs, label)
                    loss_dice_val = dice_loss(outputs, label, softmax=True)
                    loss = 0.5 * loss_ce + 0.5 * loss_dice_val
                    val_loss_sum += loss.item()

                    batch_mdice = calculate_mdice(outputs, label, num_classes)
                    val_mdice_sum += batch_mdice

            avg_val_loss = val_loss_sum / len(val_loader)
            avg_val_mdice = val_mdice_sum / len(val_loader)

            scheduler.step()

            with open(log_file, 'a') as f:
                f.write(f"{epoch+1},{avg_train_loss:.5f},{avg_val_loss:.5f},{avg_val_mdice:.5f}\n")

            if avg_val_mdice > best_mdice:
                best_mdice = avg_val_mdice
                epochs_no_improve = 0
                torch.save(model.state_dict(), os.path.join(current_save_dir, 'best_model.pth'))
                save_msg = " Best mDice!"
            else:
                epochs_no_improve += 1
                save_msg = f" No improve ({epochs_no_improve}/{PATIENCE})"

            torch.save(model.state_dict(), os.path.join(current_save_dir, 'last_model.pth'))

            print(f" [Ep {epoch+1}] T_Loss: {avg_train_loss:.4f} | V_Loss: {avg_val_loss:.4f} | V_mDice: {avg_val_mdice:.4f} | {save_msg}", flush=True)

            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping {dataset_name} after {PATIENCE} epochs without validation improvement.", flush=True)
                break

        print(f"\n {dataset_name} Finished. Best mDice: {best_mdice:.4f}", flush=True)

    except Exception as e:
        print(f"\n Failed task {dataset_name}: {e}", flush=True)
        traceback.print_exc()

    del model, optimizer, scaler, train_loader, val_loader
    torch.cuda.empty_cache()
    gc.collect()


def parse_args():
    parser = argparse.ArgumentParser(description="Train Swin-Unet on one dataset.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--pretrained-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default=Args.cfg)
    parser.add_argument("--device", default=DEVICE)
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    BASE_SAVE_DIR = cli_args.output_dir
    PRETRAINED_PATH = cli_args.pretrained_checkpoint
    DEVICE = cli_args.device
    args.cfg = cli_args.config
    args.output_dir = BASE_SAVE_DIR
    dataset_name = cli_args.dataset_name or os.path.basename(
        os.path.normpath(cli_args.dataset_root)
    )

    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)

    print("Swin-Unet training started.", flush=True)
    run_training_for_dataset(
        dataset_name, cli_args.dataset_root, cli_args.num_classes
    )

    print(f"\n All tasks finished!", flush=True)
