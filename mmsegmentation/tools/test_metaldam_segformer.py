from test_segformer_dataset import run


run({
    'dataset_name': 'MetalDAM',
    'config': 'configs/segformer/segformer_mit-b0_1xb2-2k_metaldam-512x512.py',
    'work_dir': 'work_dirs/segformer_mit-b0_1xb2-2k_metaldam-512x512',
    'img_dir': None,
    'label_dir': None,
    'result_root': 'work_dirs/results/MetalDAM_SegFormer',
    'num_classes': 5,
    'batch_size': 16,
    'image_exts': ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'],
    'label_exts': ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'],
})
