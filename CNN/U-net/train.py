import os
import sys
import argparse

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import segmentation_models_pytorch as smp
import albumentations as albu
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
plt.switch_backend('agg')

try:
    from torch.amp import autocast, GradScaler
except ImportError:
    from torch.cuda.amp import autocast, GradScaler


DATA_DIR = None
SAVE_DIR = None
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

BATCH_SIZE = 32
NUM_WORKERS = 8
EPOCHS = 500
LEARNING_RATE = 3e-4
INPUT_SIZE = 512
DO_SANITY_CHECK = True

ENCODER = 'resnet50'
ENCODER_WEIGHTS = 'imagenet'

PATIENCE = 50

class UniversalDataset(Dataset):
    def __init__(self, root_dir, mode='train', augmentation=None, preprocessing=None):
        self.images_dir = os.path.join(root_dir, mode, 'images')
        self.masks_dir = os.path.join(root_dir, mode, 'masks')

        valid_exts = ('.jpg', '.png', '.tif', '.bmp', '.jpeg')
        self.image_names = sorted([f for f in os.listdir(self.images_dir) if f.lower().endswith(valid_exts)])
        self.mask_files_map = {f.lower(): f for f in os.listdir(self.masks_dir)}

        self.images_fps = []
        self.masks_fps = []


        self.match_rules = [
            lambda x: x.replace("RG", "RGMask", 1),
            lambda x: x + "_mask",
            lambda x: x + "_label",
            lambda x: x + "_seg",
            lambda x: x,
        ]

        print(f"Matching {mode} images and masks.", flush=True)
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

        print(f"Loaded {len(self.images_fps)} {mode} samples.", flush=True)
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


        mask = cv2.imread(self.masks_fps[i], -1)


        if mask.ndim == 3:
            mask = np.max(mask, axis=2)


        h_img, w_img = image.shape[:2]
        h_mask, w_mask = mask.shape[:2]
        if h_img != h_mask or w_img != w_mask:
            h_min = min(h_img, h_mask)
            w_min = min(w_img, w_mask)
            image = image[:h_min, :w_min]
            mask = mask[:h_min, :w_min]


        mask_max = mask.max()
        if mask_max > 127:

            mask = np.where(mask > 127, 1.0, 0.0).astype('float32')
        else:

            mask = np.where(mask > 0, 1.0, 0.0).astype('float32')


        mask = np.expand_dims(mask, axis=-1)

        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        return len(self.images_fps)


def get_training_augmentation():
    train_transform = [
        albu.Resize(INPUT_SIZE, INPUT_SIZE),
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.RandomRotate90(p=0.5),
        albu.ShiftScaleRotate(scale_limit=0.2, rotate_limit=30, shift_limit=0.1, p=0.5, border_mode=0),
        albu.OneOf([
            albu.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05),
            albu.GridDistortion(p=0.5),
            albu.OpticalDistortion(distort_limit=0.05, p=0.5),
        ], p=0.3),
        albu.RandomBrightnessContrast(p=0.2),
        albu.GaussNoise(p=0.2),
    ]
    return albu.Compose(train_transform)

def get_validation_augmentation():
    return albu.Compose([albu.Resize(INPUT_SIZE, INPUT_SIZE)])

def get_preprocessing(preprocessing_fn):
    return albu.Compose([
        albu.Lambda(image=preprocessing_fn),
        albu.pytorch.ToTensorV2(),
    ])


class CompoundLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice = smp.losses.DiceLoss(mode='binary')
        self.bce = smp.losses.SoftBCEWithLogitsLoss()

    def forward(self, logits, targets):
        loss_d = self.dice(logits, targets)
        loss_b = self.bce(logits, targets)
        return 0.5 * loss_d + 0.5 * loss_b


def check_sanity(dataset, save_dir):
    print("Running the dataset sanity check.", flush=True)
    idx = np.random.randint(0, len(dataset))
    image, mask = dataset[idx]

    img_vis = image.permute(1, 2, 0).cpu().numpy()
    img_vis = (img_vis - img_vis.min()) / (img_vis.max() - img_vis.min() + 1e-7)
    mask_vis = mask.squeeze().cpu().numpy()

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img_vis)
    plt.title("Input Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(mask_vis, cmap='gray', vmin=0, vmax=1)
    plt.title(f"Mask (Unique: {np.unique(mask_vis)})")
    plt.axis('off')

    save_path = os.path.join(save_dir, 'sanity_check.png')
    plt.savefig(save_path)
    print(f"Saved the sanity-check image to {save_path}", flush=True)
    plt.close()

def plot_history(history, save_path):
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(15, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['val_dice'], 'g-', label='Validation Dice')
    plt.title('Dice Score Curve')
    plt.xlabel('Epochs'); plt.ylabel('Dice'); plt.legend(); plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved the training curve to {save_path}", flush=True)
    plt.close()


def train_one_epoch(model, loader, optimizer, loss_fn, device, scaler):
    model.train()
    epoch_loss = 0
    total_batches = len(loader)


    for batch_idx, (images, masks) in enumerate(loader):
        images = images.to(device, dtype=torch.float32)
        masks = masks.to(device, dtype=torch.float32)
        if masks.ndim == 4 and masks.shape[-1] == 1:
            masks = masks.permute(0, 3, 1, 2)

        optimizer.zero_grad()
        with autocast(device_type='cuda'):
            outputs = model(images)
            loss = loss_fn(outputs, masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()


        if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == total_batches:
            print(f"   [Batch {batch_idx + 1}/{total_batches}] Loss: {loss.item():.4f}", flush=True)

    return epoch_loss / total_batches

def evaluate(model, loader, loss_fn, device):
    model.eval()
    epoch_loss = 0
    dice_score = 0
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device, dtype=torch.float32)
            masks = masks.to(device, dtype=torch.float32)
            if masks.ndim == 4 and masks.shape[-1] == 1:
                masks = masks.permute(0, 3, 1, 2)

            with autocast(device_type='cuda'):
                outputs = model(images)
                loss = loss_fn(outputs, masks)

            epoch_loss += loss.item()
            pred_prob = torch.sigmoid(outputs)
            pred = (pred_prob > 0.5).float()
            intersection = (pred * masks).sum()
            union = pred.sum() + masks.sum()
            dice = (2. * intersection) / (union + 1e-7)
            dice_score += dice.item()
    return epoch_loss / len(loader), dice_score / len(loader)


def parse_args():
    parser = argparse.ArgumentParser(description="Train binary U-Net.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--no-sanity-check", action="store_true")
    return parser.parse_args()


def main():
    global DATA_DIR, SAVE_DIR, DEVICE, NUM_WORKERS, DO_SANITY_CHECK

    args = parse_args()
    DATA_DIR = args.dataset_root
    SAVE_DIR = args.output_dir
    DEVICE = args.device
    NUM_WORKERS = args.num_workers
    DO_SANITY_CHECK = not args.no_sanity_check
    os.makedirs(SAVE_DIR, exist_ok=True)

    print("Starting binary U-Net training.", flush=True)
    print(f"Using device: {DEVICE}", flush=True)

    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        in_channels=3,
        classes=1,
        activation=None
    )
    model.to(DEVICE)

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    train_dataset = UniversalDataset(DATA_DIR, mode='train', augmentation=get_training_augmentation(), preprocessing=get_preprocessing(preprocessing_fn))
    valid_dataset = UniversalDataset(DATA_DIR, mode='val', augmentation=get_validation_augmentation(), preprocessing=get_preprocessing(preprocessing_fn))

    if DO_SANITY_CHECK:
        check_sanity(train_dataset, SAVE_DIR)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    loss_fn = CompoundLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
    scaler = GradScaler()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    history = {'train_loss': [], 'val_loss': [], 'val_dice': []}
    best_dice = 0.0


    epochs_no_improve = 0

    print(f"Starting training for up to {EPOCHS} epochs (patience: {PATIENCE}).", flush=True)

    for i in range(EPOCHS):
        current_epoch = i + 1
        print(f"\n========== Epoch {current_epoch}/{EPOCHS} [LR: {optimizer.param_groups[0]['lr']:.2e}] ==========", flush=True)

        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, DEVICE, scaler)
        valid_loss, valid_dice = evaluate(model, valid_loader, loss_fn, DEVICE)

        scheduler.step()

        history['train_loss'].append(train_loss)
        history['val_loss'].append(valid_loss)
        history['val_dice'].append(valid_dice)

        print(f" Train Loss: {train_loss:.4f} | Val Loss: {valid_loss:.4f} | Val Dice: {valid_dice:.4f}", flush=True)


        if valid_dice > best_dice:
            best_dice = valid_dice
            epochs_no_improve = 0
            save_path = os.path.join(SAVE_DIR, f'Best_Model.pth')
            torch.save(model.state_dict(), save_path)
            print(f"Saved best checkpoint to {save_path} (Dice: {best_dice:.4f}).", flush=True)
        else:
            epochs_no_improve += 1
            print(f"Validation Dice did not improve ({epochs_no_improve}/{PATIENCE}).", flush=True)

        if epochs_no_improve >= PATIENCE:
            print(f"Early stopping after {PATIENCE} epochs without improvement.", flush=True)
            break


    plot_path = os.path.join(SAVE_DIR, f'training_curve.png')
    plot_history(history, plot_path)
    print("Training completed.", flush=True)

if __name__ == '__main__':
    main()
