"""Decompress the DVC-imported Galaxy Zoo 2 raw files into data/.

Run via `uv run dvc repro` (see dvc.yaml) rather than directly.
"""

import gzip
import shutil
import zipfile
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
IMAGES_DIR = DATA_DIR / "images"


def extract_csvs() -> None:
    for src in sorted(RAW_DIR.glob("*.csv.gz")):
        dst = DATA_DIR / src.name.removesuffix(".gz")
        with gzip.open(src, "rb") as f_in, dst.open("wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        print(f"{src.name} -> {dst.relative_to(DATA_DIR.parent)}")

    mapping = RAW_DIR / "gz2_filename_mapping.csv"
    shutil.copyfile(mapping, DATA_DIR / mapping.name)


def extract_images() -> None:
    """Flatten the zip's jpgs into data/images/<asset_id>.jpg."""
    if IMAGES_DIR.exists():
        shutil.rmtree(IMAGES_DIR)
    IMAGES_DIR.mkdir()

    count = 0
    with zipfile.ZipFile(RAW_DIR / "images_gz2.zip") as zf:
        for member in zf.infolist():
            name = Path(member.filename).name
            if member.is_dir() or not name.endswith(".jpg") or name.startswith("._"):
                continue
            with zf.open(member) as f_in, (IMAGES_DIR / name).open("wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
            count += 1
    print(f"images_gz2.zip -> data/images/ ({count:,} jpgs)")


def main() -> None:
    extract_csvs()
    extract_images()


if __name__ == "__main__":
    main()
