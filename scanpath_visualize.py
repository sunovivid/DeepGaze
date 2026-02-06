#!/usr/bin/env python3
import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, UnidentifiedImageError

import deepgaze_pytorch


DEFAULT_GIF_FPS = 5.0 / 3.0
DEFAULT_PEAK_KERNEL = 3


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize DeepGaze scanpaths (DeepGaze III) or local-peak scanpaths (DeepGaze IIE)."
    )
    parser.add_argument("--model", type=str, default="iii", choices=("iii", "iie"), help="Model type: 'iii' or 'iie'.")
    parser.add_argument("--image", type=str, default=None, help="Path to the input image.")
    parser.add_argument("--image-dir", type=str, default=None, help="Directory of images to visualize recursively.")
    parser.add_argument("--centerbias", type=str, default=None, help="Optional .npy centerbias log-density file.")
    parser.add_argument("--num-points", type=int, default=12, help="Number of points (scanpath steps or local peaks).")
    parser.add_argument("--out-dir", type=str, default="scanpath_viz", help="Output directory for single image frames/GIF.")
    parser.add_argument("--out-root", type=str, default="scanpath_outputs", help="Output root for image directory runs.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sampling.")
    parser.add_argument("--device", type=str, default=None, help="Device override (e.g., cpu, cuda).")
    parser.add_argument("--initial", type=str, default=None, help="Initial fixation as 'x,y' in pixel coords.")
    parser.add_argument("--min-sep", type=float, default=64.0, help="Minimum pixel distance to show older points/labels.")
    parser.add_argument("--fps", type=float, default=DEFAULT_GIF_FPS, help="GIF playback FPS.")
    parser.add_argument("--min-sep-ratio", type=float, default=15.0, help="Minimum separation ratio (%% of diag) for sampling.")
    parser.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature for fixation selection.")
    return parser.parse_args()


def load_image(path: str):
    image = Image.open(path).convert("RGB")
    return np.asarray(image)


def image_to_tensor(image_np: np.ndarray, device: torch.device):
    tensor = torch.tensor(image_np.transpose(2, 0, 1), dtype=torch.float32, device=device)
    return tensor.unsqueeze(0)


def normalize_log_density(log_density: torch.Tensor):
    return log_density - torch.logsumexp(log_density.flatten(), dim=0)


def load_centerbias(path: str, target_hw, device: torch.device):
    if path is None:
        centerbias = torch.zeros(target_hw, dtype=torch.float32, device=device)
        return normalize_log_density(centerbias)

    centerbias_np = np.load(path)
    centerbias = torch.tensor(centerbias_np, dtype=torch.float32, device=device)
    if centerbias.shape != target_hw:
        centerbias = F.interpolate(
            centerbias.unsqueeze(0).unsqueeze(0),
            size=list(target_hw),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).squeeze(0)
    return normalize_log_density(centerbias)


def resolve_included_fixations(included_fixations):
    if isinstance(included_fixations, int):
        if included_fixations < 0:
            return [-1 - i for i in range(-included_fixations)]
        return list(range(included_fixations))
    return list(included_fixations)


def build_history(xs, ys, included_fixations):
    hist_x = []
    hist_y = []
    for idx in included_fixations:
        if idx < 0:
            if len(xs) >= abs(idx):
                hist_x.append(xs[idx])
                hist_y.append(ys[idx])
            else:
                hist_x.append(np.nan)
                hist_y.append(np.nan)
        else:
            if len(xs) > idx:
                hist_x.append(xs[idx])
                hist_y.append(ys[idx])
            else:
                hist_x.append(np.nan)
                hist_y.append(np.nan)
    return np.array(hist_x, dtype=np.float32), np.array(hist_y, dtype=np.float32)


@torch.no_grad()
def sample_scanpath(
    model,
    image_tensor,
    centerbias_tensor,
    num_points,
    initial_xy=None,
    seed=0,
    min_sep_ratio=15.0,
    temperature=0.1,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    _, _, height, width = image_tensor.shape
    diag = math.hypot(width, height)
    min_sep = max(0.0, float(min_sep_ratio) / 100.0 * diag)
    min_sep_sq = min_sep * min_sep
    max_attempts = max(10, int(num_points) * 10)

    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    xs_flat = torch.tensor(grid_x.reshape(-1), dtype=torch.float32, device=image_tensor.device)
    ys_flat = torch.tensor(grid_y.reshape(-1), dtype=torch.float32, device=image_tensor.device)
    x_grid = xs_flat.view(height, width)
    y_grid = ys_flat.view(height, width)

    xs = []
    ys = []
    log_densities = []

    included = resolve_included_fixations(model.included_fixations)

    if initial_xy is None:
        x0 = int(width // 2)
        y0 = int(height // 2)
        required_hist = 1
        if included:
            pos = [i for i in included if i >= 0]
            neg = [abs(i) for i in included if i < 0]
            required_hist = max(required_hist, (max(pos) + 1) if pos else 0, max(neg) if neg else 0)
        seed_xs = [float(x0)] * required_hist
        seed_ys = [float(y0)] * required_hist
        hist_x0, hist_y0 = build_history(seed_xs, seed_ys, included)
        x_hist_tensor0 = torch.tensor([hist_x0], dtype=torch.float32, device=image_tensor.device)
        y_hist_tensor0 = torch.tensor([hist_y0], dtype=torch.float32, device=image_tensor.device)
        log_density0 = model(image_tensor, centerbias_tensor, x_hist_tensor0, y_hist_tensor0)
        if log_density0.ndim == 4:
            log_density0 = log_density0[:, 0]
        logits0 = log_density0[0] / max(temperature, 1e-6)
        logits0 = torch.nan_to_num(logits0, nan=-1e9, posinf=0.0, neginf=-1e9)
        logits0 = logits0 - logits0.max()
        probs0 = torch.exp(logits0).flatten()
        probs0_sum = probs0.sum()
        if not torch.isfinite(probs0_sum) or probs0_sum <= 0:
            probs0 = torch.ones_like(probs0)
            probs0_sum = probs0.sum()
        probs0 = probs0 / probs0_sum
        idx0 = torch.multinomial(probs0, 1).item()
        x0 = float(xs_flat[idx0].item())
        y0 = float(ys_flat[idx0].item())
    else:
        x0, y0 = initial_xy
        hist_x0, hist_y0 = build_history([], [], included)
        x_hist_tensor0 = torch.tensor([hist_x0], dtype=torch.float32, device=image_tensor.device)
        y_hist_tensor0 = torch.tensor([hist_y0], dtype=torch.float32, device=image_tensor.device)
        log_density0 = model(image_tensor, centerbias_tensor, x_hist_tensor0, y_hist_tensor0)
        if log_density0.ndim == 4:
            log_density0 = log_density0[:, 0]

    x0 = float(np.clip(x0, 0, width - 1))
    y0 = float(np.clip(y0, 0, height - 1))
    xs.append(x0)
    ys.append(y0)
    log_densities.append(log_density0[0].detach().cpu())

    attempts = 0
    while len(xs) < num_points and attempts < max_attempts:
        attempts += 1
        hist_x, hist_y = build_history(xs, ys, included)
        x_hist_tensor = torch.tensor([hist_x], dtype=torch.float32, device=image_tensor.device)
        y_hist_tensor = torch.tensor([hist_y], dtype=torch.float32, device=image_tensor.device)

        log_density = model(image_tensor, centerbias_tensor, x_hist_tensor, y_hist_tensor)
        if log_density.ndim == 4:
            log_density = log_density[:, 0]

        log_density_step = log_density[0]
        logits = log_density_step / max(temperature, 1e-6)
        if min_sep_sq > 0 and xs:
            mask = torch.zeros_like(logits, dtype=torch.bool)
            for fx, fy in zip(xs, ys):
                dx = x_grid - fx
                dy = y_grid - fy
                mask |= (dx * dx + dy * dy) < min_sep_sq
            if torch.all(mask):
                break
            logits = logits.masked_fill(mask, float("-inf"))
        logits = torch.nan_to_num(logits, nan=-1e9, posinf=0.0, neginf=-1e9)
        logits = logits - logits.max()
        probs = torch.exp(logits).flatten()
        probs_sum = probs.sum()
        if not torch.isfinite(probs_sum) or probs_sum <= 0:
            probs = torch.ones_like(probs)
            probs_sum = probs.sum()
        probs = probs / probs_sum
        index = torch.argmax(probs).item()
        x_next = float(xs_flat[index].item())
        y_next = float(ys_flat[index].item())
        if min_sep_sq > 0:
            too_close = False
            for fx, fy in zip(xs, ys):
                dx = x_next - fx
                dy = y_next - fy
                if (dx * dx + dy * dy) < min_sep_sq:
                    too_close = True
                    break
            if too_close:
                continue

        log_densities.append(log_density_step.detach().cpu())
        xs.append(float(np.clip(x_next, 0, width - 1)))
        ys.append(float(np.clip(y_next, 0, height - 1)))

    return xs, ys, log_densities


def filter_overlapping_points(xs, ys, min_sep):
    kept = []
    for idx, (x, y) in enumerate(zip(xs, ys), start=1):
        if not kept:
            kept.append(idx)
            continue
        keep = True
        for kept_idx in kept:
            dx = xs[kept_idx - 1] - x
            dy = ys[kept_idx - 1] - y
            if (dx * dx + dy * dy) < (min_sep * min_sep):
                keep = False
                break
        if keep:
            kept.append(idx)
    return set(kept)


def select_local_peaks(
    prob: torch.Tensor,
    num_points: int,
    min_sep_px: float,
    min_sep_ratio: float,
    kernel_size: int = DEFAULT_PEAK_KERNEL,
):
    if prob.ndim == 4:
        prob = prob[0, 0]
    elif prob.ndim == 3:
        prob = prob[0]
    prob = torch.nan_to_num(prob, nan=0.0, posinf=0.0, neginf=0.0)
    if prob.numel() == 0:
        return [], [], []

    kernel_size = max(1, int(kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1
    pad = kernel_size // 2
    pooled = F.max_pool2d(prob.unsqueeze(0).unsqueeze(0), kernel_size, stride=1, padding=pad)[0, 0]
    local_mask = prob == pooled
    height, width = prob.shape
    if min_sep_px is None or min_sep_px <= 0:
        diag = math.hypot(width, height)
        min_sep_px = max(1.0, float(min_sep_ratio) / 100.0 * diag)
    radius = max(1, int(round(float(min_sep_px))))

    candidate = prob.clone()
    candidate[~local_mask] = float("-inf")

    kept_xs = []
    kept_ys = []
    kept_vals = []
    target = num_points if num_points is not None and num_points > 0 else 0
    while target == 0 or len(kept_xs) < target:
        max_val = candidate.max()
        if not torch.isfinite(max_val):
            break
        flat_idx = int(torch.argmax(candidate).item())
        y = float(flat_idx // width)
        x = float(flat_idx % width)
        kept_xs.append(x)
        kept_ys.append(y)
        kept_vals.append(float(max_val.item()))

        y0 = max(0, int(y) - radius)
        y1 = min(height - 1, int(y) + radius)
        x0 = max(0, int(x) - radius)
        x1 = min(width - 1, int(x) + radius)
        candidate[y0 : y1 + 1, x0 : x1 + 1] = float("-inf")

    return kept_xs, kept_ys, kept_vals


def render_frames(log_densities, xs, ys, image_np, out_dir, min_sep):
    import matplotlib.pyplot as plt

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    frame_paths = []
    total_steps = len(log_densities)
    for i, log_density in enumerate(log_densities):
        step_xs = xs[: i + 1]
        step_ys = ys[: i + 1]
        visible_points = filter_overlapping_points(step_xs, step_ys, min_sep)
        prob = torch.exp(log_density).numpy()
        fig, ax = plt.subplots(figsize=(5.5, 4))
        ax.imshow(image_np)
        ax.imshow(prob, cmap="magma", origin="upper", alpha=0.55)

        if len(step_xs) > 1:
            ax.plot(step_xs, step_ys, "-", color="white", linewidth=1.0, alpha=0.85, zorder=5)

        current_index = i + 1
        for idx, (x, y) in enumerate(zip(step_xs, step_ys), start=1):
            if idx == current_index:
                continue
            if idx not in visible_points:
                continue
            ax.scatter(x, y, s=28, color="white", edgecolors="black", linewidths=0.6, zorder=7)
            ax.text(
                x, y, str(idx),
                color="white",
                fontsize=7,
                ha="center",
                va="center",
                zorder=8,
                bbox=dict(facecolor="black", alpha=0.55, edgecolor="none", boxstyle="round,pad=0.2"),
            )

        ax.scatter(
            step_xs[current_index - 1], step_ys[current_index - 1],
            s=180, color="yellow", edgecolors="black", linewidths=1.0, zorder=12
        )
        ax.text(
            step_xs[current_index - 1], step_ys[current_index - 1], str(current_index),
            color="black",
            fontsize=8,
            ha="center",
            va="center",
            zorder=13,
            bbox=dict(facecolor="yellow", alpha=0.9, edgecolor="black", boxstyle="round,pad=0.25"),
        )
        ax.set_title(f"Step {i + 1}/{total_steps}")
        ax.set_axis_off()
        fig.tight_layout()
        frame_path = out_path / f"step_{i + 1:02d}.png"
        fig.savefig(frame_path, dpi=200)
        frame_paths.append(frame_path)
        plt.close(fig)

    return frame_paths


def make_gif(frame_paths, out_path, fps=DEFAULT_GIF_FPS):
    from PIL import Image

    if not frame_paths:
        return
    frames = [Image.open(path).convert("P", palette=Image.ADAPTIVE) for path in frame_paths]
    fps = max(float(fps), 0.1)
    duration_ms = max(1, int(1000 / fps))
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        disposal=2,
    )


def parse_initial(value):
    if value is None:
        return None
    parts = value.split(",")
    if len(parts) != 2:
        raise ValueError("initial must be formatted as 'x,y'")
    return int(parts[0]), int(parts[1])


def iter_image_paths(root: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in exts:
            if path.name.startswith("._"):
                continue
            paths.append(path)
    return sorted(paths)


def render_static_scanpath(image_np, prob_np, xs, ys, out_path: Path):
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.imshow(image_np)
    ax.imshow(prob_np, cmap="magma", origin="upper", alpha=0.55)

    for i in range(len(xs) - 1):
        ax.annotate(
            "",
            xy=(xs[i + 1], ys[i + 1]),
            xytext=(xs[i], ys[i]),
            arrowprops=dict(arrowstyle="->", color="white", linewidth=1.2, alpha=0.85),
            zorder=6,
        )

    for idx, (x, y) in enumerate(zip(xs, ys), start=1):
        if idx == 1:
            ax.scatter(x, y, s=180, color="yellow", edgecolors="black", linewidths=1.0, zorder=12)
            ax.text(
                x, y, str(idx),
                color="black",
                fontsize=8,
                ha="center",
                va="center",
                zorder=13,
                bbox=dict(facecolor="yellow", alpha=0.9, edgecolor="black", boxstyle="round,pad=0.25"),
            )
        else:
            ax.scatter(x, y, s=28, color="white", edgecolors="black", linewidths=0.6, zorder=7)
            ax.text(
                x, y, str(idx),
                color="white",
                fontsize=7,
                ha="center",
                va="center",
                zorder=8,
                bbox=dict(facecolor="black", alpha=0.55, edgecolor="none", boxstyle="round,pad=0.2"),
            )

    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def visualize_image(
    image_path: Path,
    model,
    device,
    centerbias_path: str,
    num_points: int,
    seed: int,
    initial_xy,
    min_sep: float,
    min_sep_ratio: float,
    temperature: float,
    frames_dir: Path,
    gif_path: Path,
    fps: float,
    model_type: str,
):
    try:
        image = load_image(str(image_path))
    except (UnidentifiedImageError, OSError) as exc:
        print(f"Skipping {image_path}: {type(exc).__name__}")
        return 0
    image_tensor = image_to_tensor(image, device)
    centerbias = load_centerbias(centerbias_path, (image.shape[0], image.shape[1]), device)
    centerbias_tensor = centerbias.unsqueeze(0)

    if model_type == "iie":
        log_density = model(image_tensor, centerbias_tensor)
        if log_density.ndim == 4:
            log_density = log_density[:, 0]
        log_density = log_density[0].detach().cpu()
        prob_tensor = torch.exp(log_density)
        xs, ys, _ = select_local_peaks(
            prob_tensor,
            num_points=num_points,
            min_sep_px=min_sep,
            min_sep_ratio=min_sep_ratio,
        )
        if not xs:
            flat_index = int(torch.argmax(prob_tensor).item())
            height, width = prob_tensor.shape
            y0 = float(flat_index // width)
            x0 = float(flat_index % width)
            xs, ys = [x0], [y0]
        prob_np = prob_tensor.numpy()
        render_static_scanpath(image, prob_np, xs, ys, gif_path)
        return 1

    xs, ys, log_densities = sample_scanpath(
        model=model,
        image_tensor=image_tensor,
        centerbias_tensor=centerbias_tensor,
        num_points=num_points,
        initial_xy=initial_xy,
        seed=seed,
        min_sep_ratio=min_sep_ratio,
        temperature=temperature,
    )

    frame_paths = render_frames(log_densities, xs, ys, image, frames_dir, min_sep)
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    make_gif(frame_paths, gif_path, fps=fps)
    return len(frame_paths)


def main():
    args = parse_args()
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    if args.model == "iie":
        model = deepgaze_pytorch.DeepGazeIIE(pretrained=True).to(device)
    else:
        model = deepgaze_pytorch.DeepGazeIII(pretrained=True).to(device)
    model.eval()

    initial_xy = parse_initial(args.initial)

    if args.image_dir:
        image_root = Path(args.image_dir)
        out_root = Path(args.out_root)
        paths = iter_image_paths(image_root)
        if not paths:
            raise SystemExit(f"No images found under {image_root}")
        for image_path in paths:
            rel = image_path.relative_to(image_root)
            rel_dir = rel.parent
            stem = image_path.stem
            frames_dir = out_root / rel_dir / stem
            gif_suffix = ".png" if args.model == "iie" else ".gif"
            gif_path = out_root / rel_dir / f"{stem}{gif_suffix}"
            count = visualize_image(
                image_path=image_path,
                model=model,
                device=device,
                centerbias_path=args.centerbias,
                num_points=args.num_points,
                seed=args.seed,
                initial_xy=initial_xy,
                min_sep=args.min_sep,
                min_sep_ratio=args.min_sep_ratio,
                temperature=args.temperature,
                frames_dir=frames_dir,
                gif_path=gif_path,
                fps=args.fps,
                model_type=args.model,
            )
            if args.model == "iie":
                print(f"{image_path} -> {gif_path}")
            else:
                print(f"{image_path} -> {gif_path} ({count} frames)")
        return

    if not args.image:
        raise SystemExit("Either --image or --image-dir is required.")

    out_dir = Path(args.out_dir)
    gif_name = "scanpath.png" if args.model == "iie" else "scanpath.gif"
    gif_path = out_dir / gif_name
    count = visualize_image(
        image_path=Path(args.image),
        model=model,
        device=device,
        centerbias_path=args.centerbias,
        num_points=args.num_points,
        seed=args.seed,
        initial_xy=initial_xy,
        min_sep=args.min_sep,
        min_sep_ratio=args.min_sep_ratio,
        temperature=args.temperature,
        frames_dir=out_dir,
        gif_path=gif_path,
        fps=args.fps,
        model_type=args.model,
    )

    if args.model == "iie":
        print(f"Saved scanpath image to {gif_path}")
    else:
        print(f"Saved {count} frames to {out_dir}")
        print(f"Saved GIF to {gif_path}")


if __name__ == "__main__":
    main()
