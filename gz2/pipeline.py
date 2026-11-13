"""Framework-neutral input pipeline shared by every model in the CNN vs ViT comparison.

Every model reads the same saved split and sees identical preprocessing, so the only thing that
differs between experiments is the architecture.
"""

import numpy as np
import pandas as pd
import tensorflow as tf

from gz2.data import IMAGES_DIR, PROCESSED_DIR, WINDING_COLS

SPLIT_PATH = PROCESSED_DIR / "gz2_bar_winding_v1.csv"

# tf.data is only used for fast image decoding on the CPU. Hide any GPU from TensorFlow so it
# doesn't reserve memory that JAX needs.
tf.config.set_visible_devices([], "GPU")


def load_split(path=SPLIT_PATH) -> pd.DataFrame:
    """The labelled galaxies with their split assignment and image paths."""
    split = pd.read_csv(path)
    split["image_path"] = IMAGES_DIR.as_posix() + "/" + split["asset_id"].astype(int).astype(str) + ".jpg"
    return split


def load_image(path: tf.Tensor, img_size: int) -> tf.Tensor:
    """Decode, centre-crop 424 -> 212 px (the galaxy is always centred) and resize."""
    image = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
    image = tf.image.central_crop(image, 0.5)
    return tf.image.resize(image, (img_size, img_size), antialias=True)  # float32 in [0, 255]


def augment(image: tf.Tensor) -> tf.Tensor:
    """Flips and 90-degree rotations only: morphology doesn't depend on orientation, and colour
    tracks morphology, so there's no hue or brightness jitter."""
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    return tf.image.rot90(image, k=tf.random.uniform([], 0, 4, dtype=tf.int32))


def make_dataset(
    frame: pd.DataFrame,
    *,
    training: bool,
    img_size: int = 128,
    batch_size: int = 64,
    seed: int = 0,
) -> tf.data.Dataset:
    """Batches of dicts: image (B, H, W, 3) in [0, 255], bar (B,), bar_w (B,), winding (B, 3),
    winding_w (B,), dr7objid (B,). A weight of 0 masks a missing label.

    Training batches are shuffled, augmented and drop the last partial batch (fixed shapes, so
    the jitted train step compiles once). Evaluation keeps every galaxy.
    """
    ds = tf.data.Dataset.from_tensor_slices({
        "image_path": frame["image_path"].to_numpy(),
        "bar": frame["bar_label"].to_numpy(np.float32),
        "bar_w": frame["bar_weight"].to_numpy(np.float32),
        "winding": frame[WINDING_COLS].to_numpy(np.float32),
        "winding_w": frame["winding_weight"].to_numpy(np.float32),
        "dr7objid": frame["dr7objid"].to_numpy(np.int64),
    })
    if training:
        ds = ds.shuffle(len(frame), seed=seed, reshuffle_each_iteration=True)

    def prepare(example: dict) -> dict:
        image = load_image(example.pop("image_path"), img_size)
        example["image"] = augment(image) if training else image
        return example

    return (
        ds.map(prepare, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size, drop_remainder=training)
        .prefetch(tf.data.AUTOTUNE)
    )
