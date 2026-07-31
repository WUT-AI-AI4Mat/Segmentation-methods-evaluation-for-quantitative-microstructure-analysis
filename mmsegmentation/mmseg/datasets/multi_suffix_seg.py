# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import Sequence

import mmengine
import mmengine.fileio as fileio

from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class MultiSuffixSegDataset(BaseSegDataset):
    """Semantic segmentation dataset that matches images and masks by stem.

    This keeps the standard MMSeg data format but allows mixed image or mask
    extensions in one split, e.g. ``.jpg/.png/.tif`` images and ``.png/.tif``
    masks.
    """

    def __init__(self,
                 img_suffixes: Sequence[str] = ('.jpg', '.png', '.tif'),
                 seg_map_suffixes: Sequence[str] = ('.png', '.tif'),
                 **kwargs):
        self.img_suffixes = tuple(img_suffixes)
        self.seg_map_suffixes = tuple(seg_map_suffixes)
        super().__init__(
            img_suffix=self.img_suffixes[0],
            seg_map_suffix=self.seg_map_suffixes[0],
            **kwargs)

    def _find_seg_map(self, ann_dir: str, stem: str):
        for suffix in self.seg_map_suffixes:
            seg_map = stem + suffix
            seg_path = osp.join(ann_dir, seg_map)
            if fileio.exists(seg_path, backend_args=self.backend_args):
                return seg_map
        return stem + self.seg_map_suffixes[0]

    def load_data_list(self):
        data_list = []
        img_dir = self.data_prefix.get('img_path', None)
        ann_dir = self.data_prefix.get('seg_map_path', None)

        if not osp.isdir(self.ann_file) and self.ann_file:
            assert osp.isfile(self.ann_file), (
                f'Failed to load `ann_file` {self.ann_file}')
            lines = mmengine.list_from_file(
                self.ann_file, backend_args=self.backend_args)
            for line in lines:
                stem = line.strip()
                img_name = None
                for suffix in self.img_suffixes:
                    candidate = stem + suffix
                    if fileio.exists(
                            osp.join(img_dir, candidate),
                            backend_args=self.backend_args):
                        img_name = candidate
                        break
                if img_name is None:
                    img_name = stem + self.img_suffixes[0]
                data_info = dict(img_path=osp.join(img_dir, img_name))
                if ann_dir is not None:
                    data_info['seg_map_path'] = osp.join(
                        ann_dir, self._find_seg_map(ann_dir, stem))
                data_info['label_map'] = self.label_map
                data_info['reduce_zero_label'] = self.reduce_zero_label
                data_info['seg_fields'] = []
                data_list.append(data_info)
        else:
            for img in fileio.list_dir_or_file(
                    dir_path=img_dir,
                    list_dir=False,
                    suffix=self.img_suffixes,
                    recursive=True,
                    backend_args=self.backend_args):
                stem = osp.splitext(img)[0]
                data_info = dict(img_path=osp.join(img_dir, img))
                if ann_dir is not None:
                    data_info['seg_map_path'] = osp.join(
                        ann_dir, self._find_seg_map(ann_dir, stem))
                data_info['label_map'] = self.label_map
                data_info['reduce_zero_label'] = self.reduce_zero_label
                data_info['seg_fields'] = []
                data_list.append(data_info)
            data_list = sorted(data_list, key=lambda x: x['img_path'])

        return data_list
