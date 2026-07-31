import sys
import os
import argparse
import cv2
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import segmentation_models_pytorch as smp
import albumentations as albu
from albumentations.pytorch import ToTensorV2


current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
parent_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(parent_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)


print(f" 脚本位置: {current_dir}")
print(f" 项目根目录 (已挂载): {project_root}")


try:
    from Myutils.visualizer import Visualizer
    from Myutils.metrics import Metric
    print(" 成功导入: Myutils")
except ImportError as e:
    print(f" 导入失败: {e}")
    sys.exit(1)


IMG_DIR = None
LBL_DIR = None
RESULT_ROOT = None
MODEL_PATH = None


ENCODER = 'resnet50'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INPUT_SIZE = 512


def cv_imread(file_path, flags=cv2.IMREAD_COLOR):
    """Read an image from paths that OpenCV may not handle directly."""
    if not os.path.exists(file_path): return None
    img = None
    try:
        img_array = np.fromfile(file_path, dtype=np.uint8)
        img = cv2.imdecode(img_array, flags)
    except: pass
    if img is None: return None


    if flags == cv2.IMREAD_GRAYSCALE and img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif flags == cv2.IMREAD_COLOR and img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img

def find_label_file(img_name, label_dir):
    """Find the label file associated with an image."""
    base_name = os.path.splitext(img_name)[0]

    target_name = base_name.replace("RG", "RGMask")


    candidates = [base_name, target_name, base_name + "_mask"]

    exts = ['.png', '.jpg', '.tif', '.bmp']
    for t in candidates:
        for ext in exts:
            path = os.path.join(label_dir, t + ext)
            if os.path.exists(path): return path
    return None

def get_preprocessing():
    """ U-Net 预处理: Resize -> Normalize -> Tensor """
    return albu.Compose([
        albu.Resize(INPUT_SIZE, INPUT_SIZE),
        albu.Normalize(),
        ToTensorV2()
    ])


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate binary U-Net.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default=DEVICE)
    return parser.parse_args()


def main():
    global IMG_DIR, LBL_DIR, RESULT_ROOT, MODEL_PATH, DEVICE

    args = parse_args()
    IMG_DIR = os.path.join(args.dataset_root, "test", "images")
    LBL_DIR = os.path.join(args.dataset_root, "test", "masks")
    RESULT_ROOT = args.output_dir
    MODEL_PATH = args.checkpoint
    DEVICE = args.device

    print("Starting binary U-Net evaluation.")
    print(f" 结果保存至: {RESULT_ROOT}")


    save_plot_dir = os.path.join(RESULT_ROOT, "plots")
    excel_path = os.path.join(RESULT_ROOT, "metrics_summary.xlsx")
    os.makedirs(save_plot_dir, exist_ok=True)


    print(f" 加载模型: {ENCODER} ...")
    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation='sigmoid'
    )

    if os.path.exists(MODEL_PATH):
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()
        print(" 权重加载成功")
    else:
        print(f" 错误: 找不到权重文件 {MODEL_PATH}")
        return


    valid_exts = ('.jpg', '.jpeg', '.png', '.tif', '.bmp')
    img_files = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(valid_exts)]
    print(f" 找到 {len(img_files)} 张图片")

    all_metrics = []
    preprocessing_fn = get_preprocessing()


    for img_file in tqdm(img_files, desc="Processing"):
        img_path = os.path.join(IMG_DIR, img_file)

        try:

            image_raw = cv_imread(img_path, cv2.IMREAD_COLOR)
            if image_raw is None: continue
            image_rgb_raw = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)


            h_raw, w_raw = image_rgb_raw.shape[:2]


            label_path = find_label_file(img_file, LBL_DIR)
            gt_label = None
            if label_path:
                gt_label = cv_imread(label_path, cv2.IMREAD_UNCHANGED)

                if gt_label is not None and gt_label.ndim == 3:
                    gt_label = cv2.cvtColor(gt_label, cv2.COLOR_BGR2GRAY)

                gt_label = np.where(gt_label > 0, 1, 0).astype(np.uint8)


            sample = preprocessing_fn(image=image_rgb_raw)
            input_tensor = sample['image'].unsqueeze(0).to(DEVICE)


            with torch.no_grad():
                output = model(input_tensor)

                pred_mask_512 = output.squeeze().cpu().numpy()


            pred_mask_prob = cv2.resize(pred_mask_512, (w_raw, h_raw), interpolation=cv2.INTER_LINEAR)


            pred_mask_binary = (pred_mask_prob > args.threshold).astype(np.uint8)


            if gt_label is not None:

                if gt_label.shape != pred_mask_binary.shape:
                    gt_label = cv2.resize(gt_label, (w_raw, h_raw), interpolation=cv2.INTER_NEAREST)


                current_metric = Metric.compute_all(gt_label, pred_mask_binary)
                current_metric['filename'] = img_file
                all_metrics.append(current_metric)


            save_path = os.path.join(save_plot_dir, f"{os.path.splitext(img_file)[0]}_unet.png")


            Visualizer.show_result(
                model_name="U-Net (ResNet50)",
                image=image_rgb_raw,
                gt_label=gt_label,
                pred_result=pred_mask_binary,
                save_path=save_path,
                save_separately=0
            )


            plt.close('all')

        except Exception as e:
            print(f"\n 处理出错 {img_file}: {e}")
            import traceback
            traceback.print_exc()
            continue


    print("\n 正在导出 Excel 报告...")
    if len(all_metrics) > 0:
        df = pd.DataFrame(all_metrics)
        cols = ['filename'] + [c for c in df.columns if c != 'filename']
        df = df[cols]


        mean_row = df.select_dtypes(include=[np.number]).mean()
        mean_row['filename'] = 'AVERAGE'
        df_final = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)

        df_final.to_excel(excel_path, index=False)
        print(f" Excel 已保存至: {excel_path}")
        print(f"   平均指标 (U-Net):\n{mean_row}")
    else:
        print(" 未生成有效数据 (可能是没找到标签)。")

    print(" U-Net 测试结束！")

if __name__ == "__main__":
    main()
