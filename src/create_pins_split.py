import argparse
import os
import random
import shutil
from pathlib import Path

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        if dst.exists():
            dst.unlink()
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def collect_images(class_dir: Path):
    files = [
        path for path in class_dir.iterdir() if path.is_file() and path.suffix.lower() in VALID_EXTS
    ]
    return sorted(files)


def main():
    parser = argparse.ArgumentParser(description="Create a deterministic train/test split for PinsFaceRecognition.")
    parser.add_argument("--raw-root", default="105_classes_pins_dataset")
    parser.add_argument("--out-root", default="105_classes_pins_dataset_split")
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    out_root = Path(args.out_root)
    train_root = out_root / "train"
    test_root = out_root / "test"

    if not raw_root.is_dir():
        raise FileNotFoundError(f"Raw dataset root not found: {raw_root}")

    if out_root.exists():
        if not args.force:
            raise FileExistsError(
                f"Output split root already exists: {out_root}. Re-run with --force to rebuild it."
            )
        shutil.rmtree(out_root)

    rng = random.Random(args.seed)
    class_dirs = sorted(path for path in raw_root.iterdir() if path.is_dir())
    if not class_dirs:
        raise RuntimeError(f"No class directories found under: {raw_root}")

    total_train = 0
    total_test = 0
    for class_dir in class_dirs:
        images = collect_images(class_dir)
        if len(images) < 2:
            raise RuntimeError(
                f"Class '{class_dir.name}' has only {len(images)} image(s); at least 2 are required for a train/test split."
            )

        shuffled = images[:]
        rng.shuffle(shuffled)
        test_count = max(1, int(round(len(shuffled) * args.test_ratio)))
        test_count = min(test_count, len(shuffled) - 1)
        test_files = shuffled[:test_count]
        train_files = shuffled[test_count:]

        if not train_files or not test_files:
            raise RuntimeError(f"Invalid split for class '{class_dir.name}'.")

        for src in train_files:
            link_or_copy(src, train_root / class_dir.name / src.name)
        for src in test_files:
            link_or_copy(src, test_root / class_dir.name / src.name)

        total_train += len(train_files)
        total_test += len(test_files)

    print(f"Created Pins split at {out_root}")
    print(f"Classes: {len(class_dirs)}")
    print(f"Train images: {total_train}")
    print(f"Test images: {total_test}")
    print(f"Seed: {args.seed}")
    print(f"Test ratio: {args.test_ratio}")


if __name__ == "__main__":
    main()
