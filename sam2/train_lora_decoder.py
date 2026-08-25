import argparse
import os
import sys
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

try:
    from sam2.build_sam import build_sam2
    from peft import LoraConfig, get_peft_model
    from segment_anything.utils.transforms import ResizeLongestSide
    print("Loaded SAM2 and PEFT.")
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

NUM_EPOCHS = 200
LEARNING_RATE = 1e-4
MIN_LR = 1e-6
WEIGHT_DECAY = 1e-4
MAX_PROMPTS_PER_IMAGE = 32
PATIENCE = 50


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune SAM2 with LoRA and the mask decoder.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default=MODEL_CFG)
    parser.add_argument("--device", default=DEVICE)
    return parser.parse_args()

def cv_imread(file_path, flags=cv2.IMREAD_COLOR):
    if not os.path.exists(file_path): return None
    try:
        img_array = np.fromfile(file_path, dtype=np.uint8)
        img = cv2.imdecode(img_array, flags)
    except: return None
    if img is None: return None
    if flags == cv2.IMREAD_GRAYSCALE and img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif flags == cv2.IMREAD_COLOR and img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img

class GrainSAM2Dataset(Dataset):
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

        instances = np.unique(label)
        instances = instances[instances > 0]

        points_list = []
        gt_masks_list = []

        if len(instances) > MAX_PROMPTS_PER_IMAGE:
            instances = np.random.choice(instances, MAX_PROMPTS_PER_IMAGE, replace=False)

        for inst_id in instances:
            y_coords, x_coords = np.where(label == inst_id)
            if len(y_coords) < 10: continue

            rand_idx = np.random.randint(len(y_coords))
            pt_x, pt_y = x_coords[rand_idx], y_coords[rand_idx]

            pt_transformed = self.transform.apply_coords(np.array([[pt_x, pt_y]]), original_size)
            points_list.append(pt_transformed[0])
            gt_masks_list.append((label == inst_id).astype(np.float32))

        input_image = self.transform.apply_image(image_rgb)
        input_image_torch = torch.as_tensor(input_image).permute(2, 0, 1).contiguous()

        padh = self.target_size - input_image_torch.shape[1]
        padw = self.target_size - input_image_torch.shape[2]
        input_image_torch = F.pad(input_image_torch, (0, padw, 0, padh))

        return {
            "image": input_image_torch.float(),
            "points": torch.tensor(np.array(points_list), dtype=torch.float32),
            "masks": torch.tensor(np.array(gt_masks_list), dtype=torch.float32),
            "original_size": original_size
        }

def custom_collate(batch):
    return batch

def calc_loss(pred_masks, true_masks):
    if pred_masks.dim() == 3: pred_masks = pred_masks.unsqueeze(1)
    if true_masks.dim() == 3: true_masks = true_masks.unsqueeze(1)
    bce_loss = F.binary_cross_entropy_with_logits(pred_masks, true_masks)
    pred_probs = torch.sigmoid(pred_masks)
    smooth = 1e-5
    intersection = (pred_probs * true_masks).sum(dim=(2, 3))
    union = pred_probs.sum(dim=(2, 3)) + true_masks.sum(dim=(2, 3))
    dice_loss = 1.0 - (2. * intersection + smooth) / (union + smooth)
    return bce_loss + dice_loss.mean()

def forward_pass(data, model, is_train=True):
    images = data["image"].unsqueeze(0).to(DEVICE, non_blocking=True)
    points = data["points"].to(DEVICE, non_blocking=True)
    gt_masks = data["masks"].unsqueeze(1).to(DEVICE, non_blocking=True)
    original_size = data["original_size"]

    if len(points) == 0: return None

    pixel_mean = torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1).to(DEVICE)
    pixel_std = torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1).to(DEVICE)
    images = (images - pixel_mean) / pixel_std

    labels = torch.ones(points.shape[0], dtype=torch.int, device=DEVICE)
    points_tuple = (points.unsqueeze(1), labels.unsqueeze(1))

    with torch.set_grad_enabled(is_train):
        image_features_dict = model.forward_image(images)
        vision_features = image_features_dict["vision_features"]
        backbone_fpn = image_features_dict["backbone_fpn"]

    with torch.no_grad():
        sparse_embeddings, dense_embeddings = model.sam_prompt_encoder(
            points=points_tuple, boxes=None, masks=None
        )

    vision_features_repeat = vision_features.repeat(points.shape[0], 1, 1, 1)
    high_res_features_repeat = [feat.repeat(points.shape[0], 1, 1, 1) for feat in backbone_fpn[:2]]

    with torch.set_grad_enabled(is_train):
        low_res_masks, iou_preds, _, _ = model.sam_mask_decoder(
            image_embeddings=vision_features_repeat,
            image_pe=model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            repeat_image=False,
            high_res_features=high_res_features_repeat
        )

        h = int(original_size[0])
        w = int(original_size[1])

        pred_masks = F.interpolate(
            low_res_masks,
            size=(h, w),
            mode="bilinear",
            align_corners=False
        )

        loss_mask = calc_loss(pred_masks, gt_masks)

        with torch.no_grad():
            pred_binary = (pred_masks > 0.0).float()
            intersection = (pred_binary * gt_masks).sum(dim=(2, 3))
            union = pred_binary.sum(dim=(2, 3)) + gt_masks.sum(dim=(2, 3)) - intersection
            true_iou = (intersection + 1e-5) / (union + 1e-5)

        loss_iou = F.mse_loss(iou_preds, true_iou)

        loss = loss_mask + loss_iou

    return loss

def main():
    global TRAIN_IMG_DIR, TRAIN_LBL_DIR, VAL_IMG_DIR, VAL_LBL_DIR
    global SAVE_DIR, CHECKPOINT_PATH, MODEL_CFG, DEVICE

    args = parse_args()
    TRAIN_IMG_DIR = os.path.join(args.dataset_root, "train", "images")
    TRAIN_LBL_DIR = os.path.join(args.dataset_root, "train", "masks")
    VAL_IMG_DIR = os.path.join(args.dataset_root, "val", "images")
    VAL_LBL_DIR = os.path.join(args.dataset_root, "val", "masks")
    SAVE_DIR = args.output_dir
    CHECKPOINT_PATH = args.checkpoint
    MODEL_CFG = args.config
    DEVICE = args.device
    os.makedirs(SAVE_DIR, exist_ok=True)

    print("Starting SAM2 LoRA and mask-decoder fine-tuning.")
    is_terminal = sys.stdout.isatty()

    train_dataset = GrainSAM2Dataset(TRAIN_IMG_DIR, TRAIN_LBL_DIR, is_train=True)
    val_dataset = GrainSAM2Dataset(VAL_IMG_DIR, VAL_LBL_DIR, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=custom_collate, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate, num_workers=2)

    model = build_sam2(MODEL_CFG, CHECKPOINT_PATH, device=DEVICE)

    for param in model.parameters(): param.requires_grad = False

    lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["qkv", "proj"], lora_dropout=0.05, bias="none")
    model.image_encoder = get_peft_model(model.image_encoder, lora_config)
    model.to(DEVICE)

    for param in model.sam_prompt_encoder.parameters(): param.requires_grad = False

    for param in model.sam_mask_decoder.parameters(): param.requires_grad = True
    print("Training image-encoder LoRA parameters and the mask decoder.")

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
            if loss is None: continue

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
                if v_loss is not None:
                    val_loss += v_loss.item()

        avg_val_loss = val_loss / len(val_loader)

        status_msg = f"Epoch {epoch+1}/{NUM_EPOCHS} | Train loss: {avg_train_loss:.4f} | Val loss: {avg_val_loss:.4f}"

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
            status_msg += " | saved best checkpoint"

            save_path = os.path.join(SAVE_DIR, "sam2_lora_decoder_best.pth")
            trainable_state_dict = {k: v.cpu() for k, v in model.named_parameters() if v.requires_grad}
            torch.save(trainable_state_dict, save_path)
        else:
            early_stop_counter += 1
            status_msg += f" | early stopping: {early_stop_counter}/{PATIENCE}"

        print(status_msg)

        if early_stop_counter >= PATIENCE:
            print(f"Early stopping after {PATIENCE} epochs without validation improvement.")
            break

if __name__ == "__main__":
    main()
