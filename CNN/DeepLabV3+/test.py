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
import time


try:
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)
except NameError:
    current_dir = os.getcwd()

parent_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(parent_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

print(f" 当前执行路径: {current_dir}")
print(f" 识别到的项目根目录: {project_root}")

try:
    from Myutils.visualizer import Visualizer
    from Myutils.metrics import Metric
    print(" 成功导入: Myutils")
except ImportError as e:
    print(f" 导入失败: {e}")
    print(f" 当前 sys.path 中的路径搜索列表: {sys.path[:3]} ...")

IMG_DIR = None
LBL_DIR = None
RESULT_ROOT = None
MODEL_PATH = None
ENCODER = 'resnet50'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INPUT_SIZE = 512


CROP_BOTTOM_PIXELS = 0

def cv_imread(file_path, flags=cv2.IMREAD_COLOR):
    if not os.path.exists(file_path): return None
    try:
        img_array = np.fromfile(file_path, dtype=np.uint8)
        img = cv2.imdecode(img_array, flags)
        return img
    except:
        return None

def find_label_file(img_name, label_dir):
    base_name = os.path.splitext(img_name)[0]
    candidates = [
        base_name.replace("RG", "RGMask"),
        base_name,
        base_name + "_mask",
        base_name + "_label"
    ]
    exts = ['.png', '.jpg', '.tif', '.bmp']
    for t in candidates:
        for ext in exts:
            path = os.path.join(label_dir, t + ext)
            if os.path.exists(path): return path
    return None

def get_preprocessing():
    return albu.Compose([
        albu.Resize(INPUT_SIZE, INPUT_SIZE),
        albu.Normalize(),
        ToTensorV2()
    ])

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate binary DeepLabV3+.")
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

    print("Starting binary DeepLabV3+ evaluation.")
    print(f" 结果保存至: {RESULT_ROOT}")

    save_plot_dir = os.path.join(RESULT_ROOT, "plots")
    os.makedirs(save_plot_dir, exist_ok=True)
    excel_path = os.path.join(RESULT_ROOT, "metrics_summary.xlsx")


    print(f" 加载模型: {ENCODER} + DeepLabV3Plus ...")


    model = smp.DeepLabV3Plus(
        encoder_name=ENCODER,
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None
    )

    if os.path.exists(MODEL_PATH):
        try:
            state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
            model.load_state_dict(state_dict)
            model.to(DEVICE)
            model.eval()
            print(" 权重加载成功")
        except Exception as e:
            print(f" 权重加载失败: {e}")
            return
    else:
        print(f" 错误: 找不到权重文件 {MODEL_PATH}")
        return


    valid_exts = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp')
    img_files = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(valid_exts)]
    print(f" 找到 {len(img_files)} 张图片")

    all_metrics = []
    preprocessing_fn = get_preprocessing()
    start_time = time.time()


    for img_file in tqdm(img_files, desc="Processing"):
        img_path = os.path.join(IMG_DIR, img_file)

        try:

            image_raw = cv_imread(img_path, cv2.IMREAD_COLOR)
            if image_raw is None:
                print(f"\n 读取失败: {img_file}")
                continue

            label_path = find_label_file(img_file, LBL_DIR)
            gt_label = None

            if label_path:
                gt_label = cv_imread(label_path, cv2.IMREAD_UNCHANGED)

                if gt_label is not None:
                    h_img, w_img = image_raw.shape[:2]
                    h_lbl, w_lbl = gt_label.shape[:2]

                    if h_img > h_lbl:
                        image_raw = image_raw[:h_lbl, :]

            image_rgb_raw = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
            h_raw, w_raw = image_rgb_raw.shape[:2]


            if gt_label is not None:
                if gt_label.ndim == 3:

                    gt_label = gt_label[:, :, 0]


                gt_label = (gt_label > 0).astype(np.uint8)


            sample = preprocessing_fn(image=image_rgb_raw)
            input_tensor = sample['image'].unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                output = model(input_tensor)


                pred_mask_512 = torch.sigmoid(output).squeeze().cpu().numpy()


            pred_probability = cv2.resize(
                pred_mask_512, (w_raw, h_raw), interpolation=cv2.INTER_LINEAR
            )
            pred_mask_instance = (pred_probability > args.threshold).astype(np.uint8)


            if gt_label is not None:
                if gt_label.shape != pred_mask_instance.shape:
                    gt_label = cv2.resize(gt_label, (w_raw, h_raw), interpolation=cv2.INTER_NEAREST)


                current_metric = Metric.compute_all(gt_label, pred_mask_instance)
                current_metric['filename'] = img_file
                all_metrics.append(current_metric)
            else:
                print(f"\n 警告: 找不到 {img_file} 的对应标签文件，跳过指标计算。")


            base_name = os.path.splitext(img_file)[0]
            raw_save_path = os.path.join(save_plot_dir, f"{base_name}_raw_mask.png")

            Visualizer.save_raw_prediction(
                image_shape=image_rgb_raw.shape,
                pred_result=pred_mask_instance,
                save_path=raw_save_path
            )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"\n 处理出错 {img_file}: {e}")
            import traceback
            traceback.print_exc()
            continue


    end_time = time.time()
    total_processing_time = end_time - start_time
    avg_time_per_img = total_processing_time / len(img_files) if len(img_files) > 0 else 0

    print(f"\n 预测完成！总耗时: {total_processing_time:.2f} 秒 (平均 {avg_time_per_img:.2f} 秒/张)")

    print("\n 正在导出 Excel 报告...")
    if len(all_metrics) > 0:
        df = pd.DataFrame(all_metrics)
        cols = ['filename'] + [c for c in df.columns if c != 'filename']
        df = df[cols]

        mean_row = df.select_dtypes(include=[np.number]).mean()
        mean_row['filename'] = 'AVERAGE'
        mean_row['Total_Time(s)'] = round(total_processing_time, 2)
        mean_row['Avg_Time/Img(s)'] = round(avg_time_per_img, 2)

        df_final = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)
        df_final.to_excel(excel_path, index=False)
        print(f" 报告已生成: {excel_path}")
        print(f" 平均 mIoU: {mean_row.get('miou', 0):.4f}")
    else:
        print("\n 警告: 没有计算出任何指标")

    print(" 测试结束！")

if __name__ == "__main__":
    main()
