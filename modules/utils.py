import numpy as np
import matplotlib.pyplot as plt

# plotting


def plot_signal_with_importance_background(
    x,
    attr_map,
    figsize=(20, 4),
    title="Signal + importance",
    normalize=True,
    cmap="Reds",
):
    x = np.array(x).squeeze()
    attr_map = np.array(attr_map).squeeze()

    if x.ndim != 1:
        raise ValueError(f"Ожидался одномерный ряд, получено x.shape={x.shape}")
    if attr_map.ndim != 1:
        raise ValueError(f"Ожидалась одномерная карта, получено attr_map.shape={attr_map.shape}")
    if len(x) != len(attr_map):
        raise ValueError(f"Длины не совпадают: len(x)={len(x)}, len(attr_map)={len(attr_map)}")

    imp = np.abs(attr_map).astype(float)

    if normalize:
        mn, mx = imp.min(), imp.max()
        if mx > mn:
            imp = (imp - mn) / (mx - mn)
        else:
            imp = np.zeros_like(imp)

    t = np.arange(len(x))

    fig, ax = plt.subplots(figsize=figsize)

    # фон по важности
    y_min, y_max = np.min(x), np.max(x)
    ax.imshow(
        imp[np.newaxis, :],
        aspect="auto",
        cmap=cmap,
        extent=[t[0], t[-1], y_min, y_max],
        origin="lower",
        alpha=0.95,
    )

    # сам ряд
    ax.plot(t, x, linewidth=2, color="white")

    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Signal")
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    return fig, ax
