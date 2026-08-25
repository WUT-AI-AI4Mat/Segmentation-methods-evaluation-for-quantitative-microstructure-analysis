import argparse
import os
import sys
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

try:
    from peft import LoraConfig, get_peft_model
except ImportError:
    print("PEFT is required. Install it with: pip install peft")
    sys.exit(1)

from segment_anything.utils.transforms import ResizeLongestSide
from segment_anything import sam_model_registry

TRAIN_IMG_DIR = None
TRAIN_LBL_DIR = None
VAL_IMG_DIR = None
VAL_LBL_DIR = None
SAVE_DIR = None

CHECKPOINT_PATH = None
MODEL_TYPE = "vit_b"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EPOCHS = 200
LR = 1e-4
MAX_PROMPTS_PER_IMAGE = 32
PATIENCE = 50


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune HQ-SAM with LoRA and the mask decoder.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default=DEVICE)
    return parser.parse_args()

class GrainPointPromptDataset(Dataset):
    def __init__(self, img_dir, lbl_dir, target_size=1024):
        self.img_dir = img_dir
        self.lbl_dir = lbl_dir
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

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_size = image.shape[:2]

        label = cv2.imread(lbl_path, cv2.IMREAD_UNCHANGED)

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

        input_image = self.transform.apply_image(image)
        input_image_torch = torch.as_tensor(input_image).permute(2, 0, 1).contiguous()

        padh = self.target_size - input_image_torch.shape[1]
        padw = self.target_size - input_image_torch.shape[2]
        input_image_torch = F.pad(input_image_torch, (0, padw, 0, padh))

        return {
            "image": input_image_torch,
            "original_size": original_size,
            "points": torch.tensor(np.array(points_list), dtype=torch.float32),
            "gt_masks": torch.tensor(np.array(gt_masks_list), dtype=torch.float32)
        }

def custom_collate(batch):
    return batch

def calc_loss(pred_masks, gt_masks):
    bce_loss = F.binary_cross_entropy_with_logits(pred_masks, gt_masks)
    pred_sigmoid = torch.sigmoid(pred_masks)
    intersection = (pred_sigmoid * gt_masks).sum(dim=(2, 3))
    union = pred_sigmoid.sum(dim=(2, 3)) + gt_masks.sum(dim=(2, 3))
    dice_loss = 1.0 - (2. * intersection + 1e-5) / (union + 1e-5)
    return bce_loss + dice_loss.mean()

def forward_pass(data, sam, is_train=True):
    images = data["image"].unsqueeze(0).to(DEVICE, non_blocking=True)
    points = data["points"].to(DEVICE, non_blocking=True)
    gt_masks = data["gt_masks"].unsqueeze(1).to(DEVICE, non_blocking=True)

    if len(points) == 0: return None

    pixel_mean = sam.pixel_mean.to(DEVICE)
    pixel_std = sam.pixel_std.to(DEVICE)
    images = (images - pixel_mean) / pixel_std

    labels = torch.ones(points.shape[0], dtype=torch.int, device=DEVICE)
    points_tuple = (points.unsqueeze(1), labels.unsqueeze(1))

    with torch.set_grad_enabled(is_train):
        encoder_out = sam.image_encoder(images)
        if isinstance(encoder_out, tuple):
            image_embeddings, interm_embeddings = encoder_out[0], encoder_out[1]
        else:
            image_embeddings = encoder_out
            if hasattr(sam.image_encoder, 'interm_features'):
                interm_embeddings = sam.image_encoder.interm_features
            else:
                interm_embeddings = sam.image_encoder.base_model.model.interm_features


    with torch.no_grad():
        sparse_emb, dense_emb = sam.prompt_encoder(
            points=points_tuple, boxes=None, masks=None
        )

    with torch.set_grad_enabled(is_train):
        low_res_masks_hq, iou_preds = sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=False,
            hq_token_only=True,
            interm_embeddings=interm_embeddings
        )

        pred_masks = F.interpolate(
            low_res_masks_hq, size=data["original_size"], mode="bilinear", align_corners=False
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
    global SAVE_DIR, CHECKPOINT_PATH, DEVICE

    args = parse_args()
    TRAIN_IMG_DIR = os.path.join(args.dataset_root, "train", "images")
    TRAIN_LBL_DIR = os.path.join(args.dataset_root, "train", "masks")
    VAL_IMG_DIR = os.path.join(args.dataset_root, "val", "images")
    VAL_LBL_DIR = os.path.join(args.dataset_root, "val", "masks")
    SAVE_DIR = args.output_dir
    CHECKPOINT_PATH = args.checkpoint
    DEVICE = args.device
    os.makedirs(SAVE_DIR, exist_ok=True)

    print("Starting HQ-SAM LoRA and mask-decoder fine-tuning.")
    is_terminal = sys.stdout.isatty()

    print(f"Loading base checkpoint: {CHECKPOINT_PATH}")
    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(DEVICE)

    lora_config = LoraConfig(
        r=8, lora_alpha=16, target_modules=["qkv", "proj"],
        lora_dropout=0.05, bias="none"
    )
    sam.image_encoder = get_peft_model(sam.image_encoder, lora_config)
    sam.to(DEVICE)

    for param in sam.prompt_encoder.parameters(): param.requires_grad = False
    for param in sam.mask_decoder.parameters(): param.requires_grad = True
    print("Training image-encoder LoRA parameters and the mask decoder.")

    train_dataset = GrainPointPromptDataset(TRAIN_IMG_DIR, TRAIN_LBL_DIR)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=custom_collate, num_workers=4, pin_memory=True)

    val_dataset = GrainPointPromptDataset(VAL_IMG_DIR, VAL_LBL_DIR)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate, num_workers=2, pin_memory=True)

    trainable_params = [p for p in sam.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=1e-4)

    best_loss = float('inf')
    early_stop_counter = 0

    for epoch in range(EPOCHS):
        sam.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", disable=not is_terminal)

        for batch in pbar:
            data = batch[0]
            loss = forward_pass(data, sam, is_train=True)
            if loss is None: continue

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            if is_terminal:
                pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        avg_train_loss = train_loss / len(train_loader)

        sam.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                data = batch[0]
                v_loss = forward_pass(data, sam, is_train=False)
                if v_loss is not None:
                    val_loss += v_loss.item()

        avg_val_loss = val_loss / len(val_loader)

        status_msg = f"Epoch {epoch+1}/{EPOCHS} | Train loss: {avg_train_loss:.4f} | Val loss: {avg_val_loss:.4f}"

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            early_stop_counter = 0
            status_msg += " | saved best checkpoint"

            save_path = os.path.join(SAVE_DIR, "hqsam_lora_decoder_best.pth")
            trainable_state_dict = {k: v.cpu() for k, v in sam.named_parameters() if v.requires_grad}
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
