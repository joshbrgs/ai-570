from pathlib import Path

import keras
from keras import layers
from keras.utils import plot_model

ARCHITECTURE_PNG = Path(__file__).parent / "seefood2.png"


def build_model() -> keras.Model:
    """SeeFood: one image in, hot-dog verdict + condiment recommendation out."""
    # ---- Single Input ----
    image_input = keras.Input(shape=(128, 128, 3), name="food_image")

    # ---- Shared trunk: both tasks learn from the same visual features ----
    x = layers.Conv2D(16, 3, activation="relu")(image_input)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(32, 3, activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    shared_features = layers.Dense(32, activation="relu", name="jian_yang_vision")(x)

    # ---- Head 1: Hot dog / Not hot dog (binary) ----
    hotdog_output = layers.Dense(1, activation="sigmoid", name="is_hotdog")(shared_features)

    # ---- Head 2: Condiment recommendation (ketchup/mustard/relish/none) ----
    condiment_output = layers.Dense(4, activation="softmax", name="condiment")(shared_features)

    # ---- Assemble the model ----
    return keras.Model(
        inputs=image_input,
        outputs=[hotdog_output, condiment_output],
        name="SeeFood_Condiment_Edition",
    )


def main() -> None:
    model = build_model()

    # summary() prints to stdout and returns None -- don't wrap it in print().
    model.summary()

    plot_model(
        model,
        to_file=ARCHITECTURE_PNG,
        show_shapes=True,
        show_dtype=False,
        show_layer_names=True,
        show_layer_activations=True,
        rankdir="TB",
        dpi=200,
    )

    if not ARCHITECTURE_PNG.exists():
        raise RuntimeError(
            f"plot_model did not write {ARCHITECTURE_PNG}. "
            "Install the graph deps: `uv add pydot` and `brew install graphviz`."
        )
    print(f"\nArchitecture diagram saved to {ARCHITECTURE_PNG}")


if __name__ == "__main__":
    main()
