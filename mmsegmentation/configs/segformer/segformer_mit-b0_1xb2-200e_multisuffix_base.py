_base_ = ['./segformer_mit-b0_8xb2-160k_ade20k-512x512.py']

crop_size = (512, 512)
dataset_type = 'MultiSuffixSegDataset'
data_preprocessor = dict(size=crop_size)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomResize',
        scale=(2048, 512),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 512), keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

train_cfg = dict(
    _delete_=True,
    type='EpochBasedTrainLoop',
    max_epochs=200,
    val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=True, begin=0, end=5),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=5,
        end=200,
        by_epoch=True)
]

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=True),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=True,
        interval=-1,
        save_best='mIoU',
        rule='greater',
        max_keep_ckpts=1,
        save_last=False))

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator
