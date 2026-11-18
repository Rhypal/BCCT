from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from torchvision import transforms

from .folder_task_dataset import (
    RemoteSensingFolderDataset,
    discover_canonical_classes,
)


@dataclass
class TaskSpec:
    name: str
    root: str
    train_split: str = 'train'
    test_split: str = 'test'


class ClassRegistry:
    def __init__(self):
        self.class_to_id: Dict[str, int] = {}
        self.id_to_class: List[str] = []

    def add_classes(self, classes: List[str]) -> List[int]:
        new_ids = []
        for c in sorted(classes):
            if c not in self.class_to_id:
                idx = len(self.id_to_class)
                self.class_to_id[c] = idx
                self.id_to_class.append(c)
                new_ids.append(idx)
        return new_ids

    def state_dict(self):
        return {'id_to_class': list(self.id_to_class)}

    def load_state_dict(self, state):
        self.id_to_class = list(state['id_to_class'])
        self.class_to_id = {c: i for i, c in enumerate(self.id_to_class)}

    def save_json(self, path):
        Path(path).write_text(json.dumps(self.state_dict(), indent=2), encoding='utf-8')


class IncrementalDataManager:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.registry = ClassRegistry()
        self.tasks = [TaskSpec(**x) for x in cfg['data']['tasks']]
        backbone_cfg = cfg.get('backbone', {})
        backbone_type = str(
            backbone_cfg.get('type', 'vit_b16_in21k')
        ).strip().lower().replace('-', '_')
        is_dinov3 = backbone_type.startswith('dinov3')
        dinov3_cfg = backbone_cfg.get('dinov3', {}) or {}
        weights_tag = str(
            dinov3_cfg.get(
                'weights_tag',
                backbone_cfg.get('weights_tag', 'SAT493M'),
            )
        ).upper()
        is_sat493m = is_dinov3 and weights_tag == 'SAT493M'
        # Preserve legacy checkpoint evaluation when its original config is
        # supplied. The updated default explicitly selects preprocessing v2.
        use_vit_checkpoint_norm = (
            not is_dinov3 and int(cfg['data'].get('preprocessing_version', 1)) >= 2
        )

        default_size = 256 if is_sat493m else 224
        size_value = cfg['data'].get('image_size')
        size = default_size if size_value is None else int(size_value)

        default_mean = (
            (0.430, 0.411, 0.296) if is_sat493m else
            ((0.5, 0.5, 0.5) if use_vit_checkpoint_norm else (0.485, 0.456, 0.406))
        )
        default_std = (
            (0.213, 0.156, 0.143) if is_sat493m else
            ((0.5, 0.5, 0.5) if use_vit_checkpoint_norm else (0.229, 0.224, 0.225))
        )
        mean_value = cfg['data'].get('mean')
        std_value = cfg['data'].get('std')
        mean = tuple(default_mean if mean_value is None else mean_value)
        std = tuple(default_std if std_value is None else std_value)
        if len(mean) != 3 or len(std) != 3:
            raise ValueError('data.mean and data.std must each contain 3 values.')

        print(
            'Input preprocessing: backbone={}, size={}, mean={}, std={}'
            .format(backbone_type, size, mean, std)
        )
        interpolation_name = str(cfg['data'].get('interpolation', 'bilinear')).lower()
        interpolation = {
            'bilinear': transforms.InterpolationMode.BILINEAR,
            'bicubic': transforms.InterpolationMode.BICUBIC,
        }.get(interpolation_name)
        if interpolation is None:
            raise ValueError('data.interpolation must be bilinear or bicubic.')
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.75, 1.0), interpolation=interpolation),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=mean,
                std=std,
            ),
        ])
        if is_sat493m:
            # Official DINOv3 SAT-493M inference preprocessing resizes directly
            # to a square resolution instead of ImageNet resize + center crop.
            test_spatial_transform = [transforms.Resize((size, size), interpolation=interpolation)]
        else:
            test_spatial_transform = [
                transforms.Resize(int(size * 256 / 224), interpolation=interpolation),
                transforms.CenterCrop(size),
            ]
        self.test_transform = transforms.Compose([
            *test_spatial_transform,
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def num_tasks(self):
        return len(self.tasks)

    def prepare_task(self, task_id: int):
        spec = self.tasks[task_id]
        train_root = Path(spec.root) / spec.train_split
        classes = discover_canonical_classes(train_root)
        old_ids = set(range(len(self.registry.id_to_class)))
        new_ids = self.registry.add_classes(classes)
        current_ids = sorted(self.registry.class_to_id[c] for c in classes)
        recurring_ids = sorted(set(current_ids) & old_ids)
        novel_ids = sorted(new_ids)
        return {
            'task_id': task_id,
            'task_name': spec.name,
            'classes': classes,
            'current_ids': current_ids,
            'recurring_ids': recurring_ids,
            'novel_ids': novel_ids,
            'old_ids_before_task': sorted(old_ids),
        }

    def recurring_novel_class_groups(self, max_task_id: int):
        """Return each task's recurring/novel global class IDs without mutation.

        Membership is fixed at the task's arrival: classes observed in an
        earlier task are recurring and first-seen classes are novel. This
        read-only reconstruction also works after loading a checkpoint whose
        registry already contains every class seen up to ``max_task_id``.
        """
        max_task_id = int(max_task_id)
        if not 0 <= max_task_id < len(self.tasks):
            raise IndexError(f'Invalid maximum task ID: {max_task_id}')
        seen_classes = set()
        groups = {}
        for task_id in range(max_task_id + 1):
            spec = self.tasks[task_id]
            classes = set(discover_canonical_classes(
                Path(spec.root) / spec.train_split,
            ))
            missing = sorted(classes.difference(self.registry.class_to_id))
            if missing:
                raise RuntimeError(
                    f'Task {task_id} classes are absent from the registry: {missing}'
                )
            recurring = sorted(classes & seen_classes)
            novel = sorted(classes - seen_classes)
            groups[task_id] = {
                'recurring_ids': [self.registry.class_to_id[name] for name in recurring],
                'novel_ids': [self.registry.class_to_id[name] for name in novel],
            }
            seen_classes.update(classes)
        return groups

    def build_dataset(self, task_id: int, split: str, train_aug: bool = False):
        spec = self.tasks[task_id]
        split_name = spec.train_split if split == 'train' else spec.test_split
        transform = self.train_transform if train_aug else self.test_transform
        return RemoteSensingFolderDataset(
            Path(spec.root) / split_name,
            class_to_id=self.registry.class_to_id,
            transform=transform,
            task_id=task_id,
            task_name=spec.name,
        )
