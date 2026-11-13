"""JAX / Flax NNX training loop shared by the CNN and the ViT.

Model contract: `model(images) -> (bar_logit, winding_logits)` with shapes (B,) and (B, 3), where
images are float32 (B, H, W, 3) in [0, 255]. Anything that follows it can be trained, evaluated
and compared with exactly the same code.
"""

import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
import tensorflow as tf
from flax import nnx

from gz2.evaluate import metrics
from gz2.pipeline import WINDING_COLS

_JAX_KEYS = ("image", "bar", "bar_w", "winding", "winding_w")  # dr7objid stays on the host (int64)


def count_params(model: nnx.Module) -> int:
    return sum(x.size for x in jax.tree.leaves(nnx.state(model, nnx.Param)))


def masked_losses(bar_logit: jax.Array, winding_logits: jax.Array, batch: dict) -> tuple[jax.Array, jax.Array]:
    """Each head's loss is averaged over the galaxies that have that label (weight 1)."""
    bar_ce = optax.sigmoid_binary_cross_entropy(bar_logit, batch["bar"])
    bar_loss = jnp.sum(bar_ce * batch["bar_w"]) / jnp.maximum(batch["bar_w"].sum(), 1.0)

    winding_pred = jax.nn.softmax(winding_logits, axis=-1)
    winding_mse = jnp.mean((winding_pred - batch["winding"]) ** 2, axis=-1)
    winding_loss = jnp.sum(winding_mse * batch["winding_w"]) / jnp.maximum(batch["winding_w"].sum(), 1.0)
    return bar_loss, winding_loss


def make_optimizer(
    model: nnx.Module, *, lr: float, total_steps: int, warmup_steps: int, weight_decay: float = 0.05
) -> nnx.Optimizer:
    """AdamW with linear warmup and cosine decay, plus gradient clipping (standard for ViTs, and
    used for the CNN too so the training recipe isn't a confound)."""
    schedule = optax.warmup_cosine_decay_schedule(0.0, lr, warmup_steps, max(total_steps, warmup_steps + 1))
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(schedule, weight_decay=weight_decay))
    return nnx.Optimizer(model, tx, wrt=nnx.Param)


def make_train_step(loss_weights: tuple[float, float]):
    bar_weight, winding_weight = loss_weights

    def loss_fn(model: nnx.Module, batch: dict):
        bar_logit, winding_logits = model(batch["image"])
        bar_loss, winding_loss = masked_losses(bar_logit, winding_logits, batch)
        return bar_weight * bar_loss + winding_weight * winding_loss, (bar_loss, winding_loss)

    @nnx.jit
    def train_step(model: nnx.Module, optimizer: nnx.Optimizer, batch: dict) -> jax.Array:
        (loss, (bar_loss, winding_loss)), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model, batch)
        optimizer.update(model, grads)
        return jnp.stack([loss, bar_loss, winding_loss])

    return train_step


@nnx.jit
def _forward(model: nnx.Module, images: jax.Array) -> tuple[jax.Array, jax.Array]:
    bar_logit, winding_logits = model(images)
    return jax.nn.sigmoid(bar_logit), jax.nn.softmax(winding_logits, axis=-1)


def predict(model: nnx.Module, ds: tf.data.Dataset) -> pd.DataFrame:
    """One row per galaxy: true labels, label weights and the model's predictions."""
    model.eval()
    parts = []
    for batch in ds.as_numpy_iterator():
        bar_prob, winding_pred = _forward(model, jnp.asarray(batch["image"]))
        part = pd.DataFrame({
            "dr7objid": batch["dr7objid"],
            "bar_true": batch["bar"],
            "bar_w": batch["bar_w"],
            "bar_prob": np.asarray(bar_prob),
            "winding_w": batch["winding_w"],
        })
        part[[f"{c}_true" for c in WINDING_COLS]] = batch["winding"]
        part[[f"{c}_pred" for c in WINDING_COLS]] = np.asarray(winding_pred)
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def fit(
    model: nnx.Module,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    *,
    epochs: int,
    lr: float,
    loss_weights: tuple[float, float] = (1.0, 0.5),
    patience: int = 3,
    warmup_epochs: int = 1,
    weight_decay: float = 0.05,
) -> tuple[nnx.Module, pd.DataFrame]:
    """Train with early stopping on validation loss. Returns the best model and per-epoch history."""
    steps_per_epoch = int(train_ds.cardinality())
    optimizer = make_optimizer(
        model,
        lr=lr,
        total_steps=epochs * steps_per_epoch,
        warmup_steps=warmup_epochs * steps_per_epoch,
        weight_decay=weight_decay,
    )
    train_step = make_train_step(loss_weights)

    best_model, best_val_loss, epochs_without_improvement, history = None, np.inf, 0, []
    for epoch in range(1, epochs + 1):
        start = time.time()
        model.train()
        step_losses = [
            train_step(model, optimizer, {k: jnp.asarray(batch[k]) for k in _JAX_KEYS})
            for batch in train_ds.as_numpy_iterator()
        ]
        train_loss, train_bar_loss, train_winding_loss = np.asarray(jnp.stack(step_losses).mean(axis=0))

        val = metrics(predict(model, val_ds), loss_weights)
        history.append({
            "epoch": epoch,
            "loss": train_loss,
            "bar_loss": train_bar_loss,
            "winding_loss": train_winding_loss,
            **{f"val_{k}": v for k, v in val.items()},
            "seconds": time.time() - start,
        })
        print(
            f"epoch {epoch:2d} | loss {train_loss:.4f} | val loss {val['loss']:.4f} | "
            f"val bar acc {val['bar_acc']:.3f} | val winding RMSE {val['winding_rmse']:.4f} | "
            f"{time.time() - start:.0f}s"
        )

        if val["loss"] < best_val_loss:
            best_model, best_val_loss, epochs_without_improvement = nnx.clone(model), val["loss"], 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping: no val-loss improvement for {patience} epochs")
                break

    best_model.eval()
    return best_model, pd.DataFrame(history)
