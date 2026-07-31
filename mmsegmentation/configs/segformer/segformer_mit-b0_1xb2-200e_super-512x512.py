import os

_base_ = ['./segformer_mit-b0_1xb2-200e_multisuffix_base.py']

data_root = os.environ['DATASET_ROOT']
metainfo = dict(
    classes=('background', 'material_1', 'material_2'),
    palette=[[0, 0, 0], [220, 20, 60], [0, 128, 255]])

model = dict(
    data_preprocessor={{_base_.data_preprocessor}},
    decode_head=dict(num_classes=3))

img_suffixes = ('.tif', '.tiff')
seg_map_suffixes = ('.tif', '.tiff')

train_dataloader = dict(
    batch_size=32,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        _delete_=True,
        type={{_base_.dataset_type}},
        data_root=data_root,
        metainfo=metainfo,
        img_suffixes=img_suffixes,
        seg_map_suffixes=seg_map_suffixes,
        reduce_zero_label=False,
        data_prefix=dict(img_path='train/images', seg_map_path='train/masks'),
        pipeline={{_base_.train_pipeline}}))

val_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        _delete_=True,
        type={{_base_.dataset_type}},
        data_root=data_root,
        metainfo=metainfo,
        img_suffixes=img_suffixes,
        seg_map_suffixes=seg_map_suffixes,
        reduce_zero_label=False,
        data_prefix=dict(img_path='val/images', seg_map_path='val/masks'),
        pipeline={{_base_.test_pipeline}}))

test_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        _delete_=True,
        type={{_base_.dataset_type}},
        data_root=data_root,
        metainfo=metainfo,
        img_suffixes=img_suffixes,
        seg_map_suffixes=seg_map_suffixes,
        reduce_zero_label=False,
        data_prefix=dict(img_path='test/images', seg_map_path='test/masks'),
        pipeline={{_base_.test_pipeline}}))
