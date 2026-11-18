from __future__ import annotations

import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import yaml


EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ALIASES_FILE = PROJECT_ROOT / "configs" / "class_aliases.yaml"


def load_dataset_aliases(aliases_file: Path, dataset_name: str) -> Dict[str, str]:
    """Load one dataset's exact raw-folder-to-canonical mapping."""
    if not aliases_file.is_file():
        raise FileNotFoundError(f"Class-alias file does not exist: {aliases_file}")
    with aliases_file.open("r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle) or {}

    all_aliases = document.get("aliases")
    if not isinstance(all_aliases, dict) or not all_aliases:
        raise ValueError(
            f"{aliases_file} must contain a non-empty top-level 'aliases' mapping."
        )
    if dataset_name not in all_aliases:
        available = ", ".join(sorted(map(str, all_aliases)))
        raise KeyError(
            f"Dataset section {dataset_name!r} is absent from {aliases_file}. "
            f"Available sections: {available}"
        )

    aliases = all_aliases[dataset_name]
    if not isinstance(aliases, dict) or not aliases:
        raise ValueError(
            f"Alias section {dataset_name!r} must be a non-empty mapping."
        )

    checked: Dict[str, str] = {}
    for raw_name, canonical_name in aliases.items():
        if not isinstance(raw_name, str) or not isinstance(canonical_name, str):
            raise TypeError(
                f"Every alias in section {dataset_name!r} must map strings to strings."
            )
        if not raw_name or raw_name != raw_name.strip():
            raise ValueError(f"Invalid raw class name: {raw_name!r}")
        if not canonical_name or canonical_name != canonical_name.strip():
            raise ValueError(f"Invalid canonical class name: {canonical_name!r}")
        checked[raw_name] = canonical_name
    return checked


def image_files(class_dir: Path) -> List[Path]:
    return sorted(
        path
        for path in class_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in EXTS
    )


def unique_destination(output_dir: Path, source: Path, raw_class: str) -> Path:
    """Preserve a basename unless a merged canonical class would overwrite it."""
    candidate = output_dir / source.name
    if not candidate.exists():
        return candidate

    candidate = output_dir / f"{raw_class}__{source.name}"
    if not candidate.exists():
        return candidate

    index = 2
    while True:
        candidate = output_dir / f"{raw_class}__{index}__{source.name}"
        if not candidate.exists():
            return candidate
        index += 1


def copy_images(
    sources: Iterable[tuple[Path, str]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for source, raw_class in sources:
        shutil.copy2(source, unique_destination(output_dir, source, raw_class))


def split_dataset(
    src: Path,
    dst: Path,
    aliases_file: Path,
    dataset_name: str,
    seed: int,
    test_ratio: float,
    overwrite: bool,
) -> None:
    src = src.resolve()
    dst = dst.resolve()
    if not src.is_dir():
        raise FileNotFoundError(f"Source dataset directory does not exist: {src}")
    if src == dst or src in dst.parents or dst in src.parents:
        raise ValueError(
            "Source and destination must be separate directories; neither may "
            "contain the other."
        )
    if not 0.0 < test_ratio < 1.0:
        raise ValueError("--test-ratio must be strictly between 0 and 1.")

    aliases = load_dataset_aliases(aliases_file.resolve(), dataset_name)
    class_dirs = sorted(path for path in src.iterdir() if path.is_dir())
    raw_classes = {path.name for path in class_dirs}
    undeclared = sorted(raw_classes.difference(aliases))
    absent = sorted(set(aliases).difference(raw_classes))
    if undeclared or absent:
        details = []
        if undeclared:
            details.append(f"source classes missing from YAML: {undeclared}")
        if absent:
            details.append(f"YAML classes missing from source: {absent}")
        raise ValueError(
            f"Class inventory mismatch for {dataset_name!r}: " + "; ".join(details)
        )

    # Merge raw folders that map to the same semantic class before splitting.
    # This gives one canonical label directory and one 80/20 split per class.
    canonical_images: Dict[str, List[tuple[Path, str]]] = defaultdict(list)
    for class_dir in class_dirs:
        images = image_files(class_dir)
        if not images:
            raise RuntimeError(f"No supported images found in class: {class_dir}")
        canonical = aliases[class_dir.name]
        canonical_images[canonical].extend(
            (image_path, class_dir.name) for image_path in images
        )

    too_small = {
        canonical: len(samples)
        for canonical, samples in canonical_images.items()
        if len(samples) < 2
    }
    if too_small:
        raise ValueError(
            "Every canonical class needs at least two images for non-empty "
            f"train/test splits; invalid classes: {too_small}"
        )

    # Do not touch an existing output until every source class and image has
    # passed validation. Replacement is possible only through an explicit flag.
    if dst.exists() and not dst.is_dir():
        raise NotADirectoryError(f"Destination exists but is not a directory: {dst}")
    if dst.exists() and any(dst.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Destination is not empty: {dst}. Use --overwrite to replace it."
            )
        shutil.rmtree(dst)

    rng = random.Random(seed)
    total_train = 0
    total_test = 0
    for canonical in sorted(canonical_images):
        samples = list(canonical_images[canonical])
        rng.shuffle(samples)
        test_num = max(1, round(len(samples) * test_ratio))
        test_num = min(test_num, len(samples) - 1)
        test_images = samples[:test_num]
        train_images = samples[test_num:]

        copy_images(train_images, dst / "train" / canonical)
        copy_images(test_images, dst / "test" / canonical)
        total_train += len(train_images)
        total_test += len(test_images)
        print(
            f"{canonical}: train={len(train_images)}, test={len(test_images)}"
        )

    print(
        f"Done: dataset={dataset_name}, classes={len(canonical_images)}, "
        f"train={total_train}, test={total_test}, output={dst}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split a raw scene dataset and materialize canonical class-folder "
            "names from configs/class_aliases.yaml."
        )
    )
    parser.add_argument(
        "--src",
        default="D:/Anaconda/code/0dataset/scene_class/RSI-CB256",
    )
    parser.add_argument(
        "--dst",
        default="D:/Anaconda/code/0dataset/scene_class/RSI2",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help=(
            "Top-level dataset section in class_aliases.yaml. Defaults to the "
            "source directory name."
        ),
    )
    parser.add_argument("--aliases-file", default=str(DEFAULT_ALIASES_FILE))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete a non-empty destination before creating the new split.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    src = Path(args.src)
    split_dataset(
        src=src,
        dst=Path(args.dst),
        aliases_file=Path(args.aliases_file),
        dataset_name=args.dataset or src.name,
        seed=args.seed,
        test_ratio=args.test_ratio,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
