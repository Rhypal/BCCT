from __future__ import annotations

from pathlib import Path
from typing import Dict
from functools import partial
import random

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def _collate(batch):
    return {
        'image': torch.stack([x['image'] for x in batch]),
        'label': torch.tensor([x['label'] for x in batch], dtype=torch.long),
        'path': [x['path'] for x in batch],
        'task_id': torch.tensor([x['task_id'] for x in batch], dtype=torch.long),
        'task_name': [x['task_name'] for x in batch],
    }


@torch.no_grad()
def _seed_worker(worker_id: int, base_seed: int):
    seed = int(base_seed) + int(worker_id)
    random.seed(seed)
    torch.manual_seed(seed)


def extract_features(
    backbone,
    dataset,
    batch_size: int,
    num_workers: int,
    device,
    amp: bool = True,
    seed: int | None = None,
):
    generator = None
    worker_init_fn = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        worker_init_fn = partial(_seed_worker, base_seed=int(seed))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=_collate,
        generator=generator,
        worker_init_fn=worker_init_fn,
    )
    feats, styles, labels, task_ids, paths = [], [], [], [], []
    for batch in tqdm(loader, desc=f'Extract {getattr(dataset, "task_name", "features")}', leave=False):
        images = batch['image'].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(amp and device.type == 'cuda')):
            f, s = backbone(images)
        feats.append(f.float().cpu())
        styles.append(s.float().cpu())
        labels.append(batch['label'])
        task_ids.append(batch['task_id'])
        paths.extend(batch['path'])
    return {
        'features': torch.cat(feats, 0),
        'styles': torch.cat(styles, 0),
        'labels': torch.cat(labels, 0),
        'task_ids': torch.cat(task_ids, 0),
        'paths': paths,
    }


def load_or_extract(
    backbone,
    dataset,
    cache_path: str | Path,
    batch_size: int,
    num_workers: int,
    device,
    amp: bool = True,
    enabled: bool = True,
    overwrite: bool = False,
    cache_metadata: dict | None = None,
    seed: int | None = None,
) -> Dict:
    cache_path = Path(cache_path)
    if enabled and cache_path.exists() and not overwrite:
        cached = torch.load(cache_path, map_location='cpu')
        if cache_metadata is None or cached.get('_cache_metadata') == cache_metadata:
            return cached
        print(f'Ignoring stale feature cache (metadata changed): {cache_path}')
    out = extract_features(
        backbone,
        dataset,
        batch_size,
        num_workers,
        device,
        amp,
        seed=seed,
    )
    if cache_metadata is not None:
        out['_cache_metadata'] = cache_metadata
    if enabled:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(out, cache_path)
    return out
