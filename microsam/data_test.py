import sys
import os
import argparse
import cv2
import numpy as np
import torch
import tifffile
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import time


current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


micro_sam_repo_path = os.path.join(current_dir, "micro_sam")

if os.path.exists(micro_sam_repo_path):
    if micro_sam_repo_path not in sys.path:
        sys.path.insert(0, micro_sam_repo_path)
    print(f" 已添加库路径: {micro_sam_repo_path}")
else:
    print(f" 警告: 找不到文件夹 {micro_sam_repo_path}")
    print(f"   请确认你的目录里有一个叫 'micro_sam' 的文件夹")

torch_em_repo_path = os.path.join(current_dir, "torch-em")

if os.path.exists(torch_em_repo_path):
    if torch_em_repo_path not in sys.path:
        sys.path.insert(0, torch_em_repo_path)
    print(f" 已添加库路径: {torch_em_repo_path}")
else:
    print(f" 警告: 找不到 'torch-em' 文件夹。如果你的环境里没装 torch_em，下一步可能会报错。")

print("-" * 30)
try:
    from Myutils.visualizer import Visualizer
    from Myutils.metrics import Metric
    print(" 成功导入 Myutils.visualizer 和 Metric")
except ImportError as e:
    print(f" 导入 Myutils 失败: {e}")
    sys.exit(1)
try:
    from micro_sam.automatic_segmentation import get_predictor_and_segmenter, automatic_instance_segmentation
    from micro_sam.util import get_device
    print(" 成功导入 micro_sam 库")
except ImportError as e:
    print(f" 导入 micro_sam 失败: {e}")
    print("    常见原因: 缺少依赖库 'torch_em'。")
    print("   如果报错说 No module named 'torch_em'，请确保你下载了 torch-em 源码并放在了 MicroSAM 目录下。")
    sys.exit(1)
print("-" * 30)


IMG_DIR = None
LBL_DIR = None
RESULT_ROOT = None
SAVE_PLOT_DIR = None
EXCEL_PATH = None


MODEL_TYPE = "vit_b_lm"
CHECKPOINT_PATH = None
DEVICE = get_device(None)

MICROSAM_PARAMS = {
    "min_size": 10,
    "center_distance_threshold": 0.5,
    "boundary_distance_threshold": 0.5,
    "ndim": 2,
    "verbose": False
}


CROP_BOTTOM_PIXELS = 0

LABEL_RULES = [
    lambda x: x,
    lambda x: x.replace("RG", "RGMask"),
    lambda x: x + "_mask",
    lambda x: x + "_label",
    lambda x: x + "_seg",
]
SUPPORTED_EXTS = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']



def cv_imread(file_path, flags=cv2.IMREAD_COLOR):
    """Read an image while preserving class indices when requested."""
    if not os.path.exists(file_path): return None

    img = None


    try:

        img_array = np.fromfile(file_path, dtype=np.uint8)
        img = cv2.imdecode(img_array, flags)
    except: pass


    if img is None and file_path.lower().endswith(('.tif', '.tiff')):
        try:
            import tifffile
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




    if flags == cv2.IMREAD_UNCHANGED or flags == -1:
        return img


    if flags == cv2.IMREAD_GRAYSCALE:
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    elif flags == cv2.IMREAD_COLOR:
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    return img

def find_label_file(img_name, label_dir):
    """Find the label file associated with an image."""
    base_name = os.path.splitext(img_name)[0]
    for rule_func in LABEL_RULES:
        try: target_name = rule_func(base_name)
        except: continue
        for ext in SUPPORTED_EXTS:
            test_path = os.path.join(label_dir, target_name + ext)
            if os.path.exists(test_path): return test_path
    return None

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the original micro-sam model.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--model-type", default=MODEL_TYPE)
    parser.add_argument("--min-size", type=int, default=MICROSAM_PARAMS["min_size"])
    parser.add_argument(
        "--center-distance-threshold",
        type=float,
        default=MICROSAM_PARAMS["center_distance_threshold"],
    )
    parser.add_argument(
        "--boundary-distance-threshold",
        type=float,
        default=MICROSAM_PARAMS["boundary_distance_threshold"],
    )
    parser.add_argument("--crop-bottom-pixels", type=int, default=CROP_BOTTOM_PIXELS)
    return parser.parse_args()


def main():
    global IMG_DIR, LBL_DIR, RESULT_ROOT, SAVE_PLOT_DIR, EXCEL_PATH
    global CHECKPOINT_PATH, MODEL_TYPE, CROP_BOTTOM_PIXELS

    args = parse_args()
    IMG_DIR = os.path.join(args.dataset_root, "test", "images")
    LBL_DIR = os.path.join(args.dataset_root, "test", "masks")
    RESULT_ROOT = args.output_dir
    SAVE_PLOT_DIR = os.path.join(RESULT_ROOT, "plots")
    EXCEL_PATH = os.path.join(RESULT_ROOT, "metrics_summary.xlsx")
    CHECKPOINT_PATH = args.checkpoint
    MODEL_TYPE = args.model_type
    CROP_BOTTOM_PIXELS = args.crop_bottom_pixels
    MICROSAM_PARAMS["min_size"] = args.min_size
    MICROSAM_PARAMS["center_distance_threshold"] = args.center_distance_threshold
    MICROSAM_PARAMS["boundary_distance_threshold"] = args.boundary_distance_threshold

    print("Starting micro-sam evaluation.")
    print(f"Model: {MODEL_TYPE}; mode: AIS")
    print(f"Crop bottom: {CROP_BOTTOM_PIXELS} px")



    MAX_MASK_AREA = 2000000
    print(f"Maximum retained mask area: {MAX_MASK_AREA} px")


    os.makedirs(SAVE_PLOT_DIR, exist_ok=True)


    print(f" 加载模型中...")
    predictor, segmenter = get_predictor_and_segmenter(
        model_type=MODEL_TYPE,
        checkpoint=CHECKPOINT_PATH,
        device=DEVICE,
        amg=False
    )


    valid_extensions = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp')
    img_files = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(valid_extensions)]
    print(f" 找到 {len(img_files)} 张图片")

    all_metrics_list = []
    start_time = time.time()


    for img_file in tqdm(img_files, desc="Processing"):
        img_path = os.path.join(IMG_DIR, img_file)


        try:

            image = cv_imread(img_path, cv2.IMREAD_COLOR)
            if image is None:
                print(f"\n 读图失败: {img_file}")
                continue
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


            h_raw_origin, w_raw_origin = image_rgb.shape[:2]


            if CROP_BOTTOM_PIXELS > 0 and h_raw_origin > CROP_BOTTOM_PIXELS:
                image_rgb = image_rgb[:-CROP_BOTTOM_PIXELS, :]


            label_path = find_label_file(img_file, LBL_DIR)
            gt_label = None
            if label_path:
                gt_label = cv_imread(label_path, cv2.IMREAD_UNCHANGED)


            if gt_label is not None:
                h_lbl_origin, w_lbl_origin = gt_label.shape[:2]


                if CROP_BOTTOM_PIXELS > 0:
                    if abs(h_lbl_origin - h_raw_origin) < 2:
                        gt_label = gt_label[:-CROP_BOTTOM_PIXELS, :]


                h_img_curr, w_img_curr = image_rgb.shape[:2]
                h_lbl_curr, w_lbl_curr = gt_label.shape[:2]

                common_h = min(h_img_curr, h_lbl_curr)
                common_w = min(w_img_curr, w_lbl_curr)

                if h_img_curr > common_h: image_rgb = image_rgb[:common_h, :]
                if h_lbl_curr > common_h: gt_label = gt_label[:common_h, :]
                if w_img_curr > common_w: image_rgb = image_rgb[:, :common_w]
                if w_lbl_curr > common_w: gt_label = gt_label[:, :common_w]


            pred_mask = automatic_instance_segmentation(
                predictor=predictor,
                segmenter=segmenter,
                input_path=image_rgb,
                **MICROSAM_PARAMS
            )


            if pred_mask is None:
                print(f"\n 警告: {img_file} 预测结果为空")
                continue



            unique_ids = np.unique(pred_mask)
            filtered_pred = np.zeros_like(pred_mask)
            for uid in unique_ids:
                if uid == 0: continue
                instance_mask = (pred_mask == uid)
                if instance_mask.sum() <= MAX_MASK_AREA:
                    filtered_pred[instance_mask] = uid
            pred_mask = filtered_pred




            if gt_label is not None:
                current_metrics = Metric.compute_all(gt_label, pred_mask)
                current_metrics['filename'] = img_file
                all_metrics_list.append(current_metrics)
            else:
                print(f"\n 警告: 找不到 {img_file} 的对应标签文件，跳过指标计算。")


            base_name = os.path.splitext(img_file)[0]
            raw_save_path = os.path.join(SAVE_PLOT_DIR, f"{base_name}_raw_mask.png")

            Visualizer.save_raw_prediction(
                image_shape=image_rgb.shape,
                pred_result=pred_mask,
                save_path=raw_save_path
            )

        except Exception as e:

            print(f"\nFailed to process {img_file}: {e}")
            import traceback
            traceback.print_exc()
            continue


        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    end_time = time.time()
    total_processing_time = end_time - start_time
    avg_time_per_img = total_processing_time / len(img_files) if len(img_files) > 0 else 0

    print(f"\n 预测完成！总耗时: {total_processing_time:.2f} 秒 (平均 {avg_time_per_img:.2f} 秒/张)")


    print("\n 正在导出 Excel 报告...")
    if len(all_metrics_list) > 0:
        df = pd.DataFrame(all_metrics_list)
        cols = ['filename'] + [c for c in df.columns if c != 'filename']
        df = df[cols]
        mean_row = df.select_dtypes(include=[np.number]).mean()


        mean_row['filename'] = 'AVERAGE'
        mean_row['Total_Time(s)'] = round(total_processing_time, 2)
        mean_row['Avg_Time/Img(s)'] = round(avg_time_per_img, 2)

        df_final = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)

        df_final.to_excel(EXCEL_PATH, index=False)
        print(f" Excel 已保存至: {EXCEL_PATH}")
    else:
        print(" 没有产生有效数据")

    print(" 测试结束！")

if __name__ == "__main__":
    main()
