from test_segformer_dataset import run


run({
    'dataset_name': 'Grain',
    'config': 'configs/segformer/segformer_mit-b0_1xb2-200e_grain-512x512.py',
    'work_dir': 'work_dirs/segformer_mit-b0_1xb2-200e_grain-512x512',
    'img_dir': None,
    'label_dir': None,
    'result_root': 'work_dirs/results/Grain_SegFormer',
    'num_classes': 2,
    'batch_size': 16,
    'image_exts': ['.png', '.jpg', '.jpeg'],
    'label_exts': ['.png'],
})
