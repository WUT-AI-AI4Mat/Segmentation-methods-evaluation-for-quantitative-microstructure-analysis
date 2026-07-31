from test_segformer_dataset import run


run({
    'dataset_name': 'Super',
    'config': 'configs/segformer/segformer_mit-b0_1xb2-200e_super-512x512.py',
    'work_dir': 'work_dirs/segformer_mit-b0_1xb2-200e_super-512x512',
    'img_dir': None,
    'label_dir': None,
    'result_root': 'work_dirs/results/Super_SegFormer',
    'num_classes': 3,
    'batch_size': 16,
    'image_exts': ['.tif', '.tiff'],
    'label_exts': ['.tif', '.tiff'],
})
