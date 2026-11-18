from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image
from torch.utils.data import Dataset

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def discover_canonical_classes(split_root: str | Path) -> List[str]:
    """Discover already-canonical class names directly from split folders."""
    split_root = Path(split_root)
    if not split_root.exists():
        raise FileNotFoundError(f'Split directory does not exist: {split_root}')
    return sorted(d.name for d in split_root.iterdir() if d.is_dir())


class RemoteSensingFolderDataset(Dataset):
    """Folder classification dataset with a global canonical class registry."""

    def __init__(
        self,
        split_root: str | Path,
        class_to_id: Dict[str, int],
        transform=None,
        task_id: int = 0,
        task_name: str = '',
    ):
        self.split_root = Path(split_root)
        self.class_to_id = class_to_id
        self.transform = transform
        self.task_id = task_id
        self.task_name = task_name
        self.samples: List[Tuple[Path, int, str]] = []

        if not self.split_root.exists():
            raise FileNotFoundError(f'Split directory does not exist: {self.split_root}')

        for class_dir in sorted([p for p in self.split_root.iterdir() if p.is_dir()]):
            canonical = class_dir.name
            if canonical not in class_to_id:
                raise KeyError(
                    f'Class {canonical!r} from {class_dir} is not in global class registry. '
                    'The train split should be registered before building datasets.'
                )
            y = class_to_id[canonical]
            for p in sorted(class_dir.rglob('*')):
                if p.is_file() and p.suffix.lower() in IMG_EXTS:
                    self.samples.append((p, y, canonical))

        if not self.samples:
            raise RuntimeError(f'No images found in {self.split_root}')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label, class_name = self.samples[index]
        with Image.open(path) as im:
            image = im.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return {
            'image': image,
            'label': label,
            'class_name': class_name,
            'path': str(path),
            'task_id': self.task_id,
            'task_name': self.task_name,
        }
