"""Load GZ2 Table 1 (Hart et al. 2016) as one row per galaxy, joined to its image file."""

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
IMAGES_DIR = ROOT / "data" / "images"
PROCESSED_DIR = ROOT / "data" / "processed"

# e.g. t03_bar_a06_bar_debiased -> t03_bar ; t10_arms_winding_a28_tight_debiased -> t10_tight
_DEBIASED = re.compile(r"^(t\d\d)_[a-z_]+?_a\d\d_(.+)_debiased$")
_COUNT = re.compile(r"^(t\d\d)_[a-z_]+?_a\d\d_.+_count$")

ID_COLUMNS = ["dr7objid", "ra", "dec", "sample", "gz2_class", "total_classifications", "total_votes"]


def morphology(gz2_class: str) -> str:
    """Collapse the GZ2 class string (e.g. 'SBc2m', 'Ei') to a coarse type."""
    if gz2_class.startswith("A"):
        return "Artifact"
    if gz2_class.startswith("E"):
        return "Smooth"
    if gz2_class.startswith("SB"):
        return "Barred spiral"
    return "Spiral"


def load_galaxies(raw_dir: Path = RAW_DIR, images_dir: Path = IMAGES_DIR) -> pd.DataFrame:
    """One row per galaxy with debiased vote fractions, per-question vote totals and image paths.

    Columns:
      - ID_COLUMNS from Table 1, plus `morphology`
      - `tNN_<answer>`: debiased vote fraction for each answer (37 columns)
      - `tNN_votes`: volunteers who answered question NN (deep questions get far fewer votes)
      - `asset_id`, `image_path`, `has_image`
    """
    hart = pd.read_csv(raw_dir / "gz2_hart16.csv.gz")
    mapping = pd.read_csv(raw_dir / "gz2_filename_mapping.csv")

    fractions = {c: f"{m.group(1)}_{m.group(2)}" for c in hart.columns if (m := _DEBIASED.match(c))}
    count_cols = [c for c in hart.columns if _COUNT.match(c)]
    votes = (
        hart[count_cols]
        .T.groupby(lambda c: _COUNT.match(c).group(1)).sum().T
        .add_suffix("_votes")
    )

    df = (
        hart[ID_COLUMNS]
        .join(hart[list(fractions)].rename(columns=fractions))
        .join(votes)
        .merge(
            mapping.rename(columns={"objid": "dr7objid"}),
            on=["dr7objid", "sample"],
            how="left",
            validate="one_to_one",
        )
    )
    df["morphology"] = df["gz2_class"].map(morphology)
    asset = df["asset_id"].astype("Int64")
    df["image_path"] = images_dir.as_posix() + "/" + asset.astype(str) + ".jpg"
    existing = {int(p.stem) for p in images_dir.glob("*.jpg")}
    df["has_image"] = asset.isin(existing).fillna(False).astype(bool)
    return df
