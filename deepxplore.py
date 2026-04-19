"""
deepxplore.py
PyTorch re-implementation of DeepXplore for CIFAR-10 / ResNet50.

Reference:
  Pei et al., "DeepXplore: Automated Whitebox Testing of Deep Learning Systems"
  SOSP 2017.  https://github.com/peikexin9/deepxplore

Key ideas implemented:
  1. Neuron Coverage  – fraction of neurons activated above a threshold.
  2. Joint Loss       – maximise disagreement between models while
                        simultaneously maximising neuron coverage.
  3. Gradient-guided  – iteratively perturb a seed image with the joint
                        gradient until the two models disagree.
"""

import os
import copy
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use("Agg")          # headless – no display needed
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────
# CIFAR-10 class names
# ─────────────────────────────────────────────────────────────
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


# ─────────────────────────────────────────────────────────────
# 1. Neuron Coverage Tracker
# ─────────────────────────────────────────────────────────────

class NeuronCoverageTracker:
    """
    Tracks which neurons have been activated (output > threshold)
    across all forward passes seen so far.

    Uses PyTorch forward hooks to intercept intermediate activations
    from every ReLU layer in the model.
    """

    def __init__(self, model: nn.Module, threshold: float = 0.5):
        self.threshold = threshold
        self.hooks: List[torch.utils.hooks.RemovableHook] = []

        # activated[layer_name][neuron_idx] = True/False
        self.activated: Dict[str, torch.Tensor] = {}
        self.total_neurons = 0

        self._register_hooks(model)

    def _register_hooks(self, model: nn.Module):
        """Attach a forward hook to every ReLU in the model."""
        for name, module in model.named_modules():
            if isinstance(module, nn.ReLU):
                # Use module's object id to guarantee uniqueness.
                # The same ReLU instance can appear under different names
                # (e.g. inplace reuse), so id() gives a stable unique key.
                unique_key = f"{name}__{id(module)}"
                hook = module.register_forward_hook(
                    self._make_hook(unique_key)
                )
                self.hooks.append(hook)

    def _make_hook(self, layer_key: str):
        def hook_fn(module, input, output):
            # output shape: (batch, C, H, W) or (batch, C)
            flat = output.detach().view(output.size(0), -1)  # (B, N)

            # A neuron is "covered" if ANY sample in the batch exceeds threshold
            newly_covered = (flat > self.threshold).any(dim=0)  # (N,)

            if layer_key not in self.activated:
                # First time seeing this layer — initialise entry
                self.activated[layer_key] = newly_covered.clone()
                self.total_neurons += newly_covered.numel()
            else:
                existing = self.activated[layer_key]
                if existing.shape == newly_covered.shape:
                    # Normal case: OR in new activations
                    self.activated[layer_key] = existing | newly_covered
                else:
                    # Shape mismatch (e.g. different spatial resolution on
                    # second call): create a separate entry with a suffix
                    fallback_key = f"{layer_key}__sz{newly_covered.numel()}"
                    if fallback_key not in self.activated:
                        self.activated[fallback_key] = newly_covered.clone()
                        self.total_neurons += newly_covered.numel()
                    else:
                        self.activated[fallback_key] = (
                            self.activated[fallback_key] | newly_covered
                        )

        return hook_fn

    def coverage(self) -> float:
        """Return fraction of neurons covered so far."""
        if self.total_neurons == 0:
            return 0.0
        covered = sum(t.sum().item() for t in self.activated.values())
        return covered / self.total_neurons

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()


# ─────────────────────────────────────────────────────────────
# 2. Joint Loss
# ─────────────────────────────────────────────────────────────

def compute_joint_loss(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    lambda_cov: float = 0.5,
) -> torch.Tensor:
    """
    Joint loss = disagreement_loss + lambda_cov * coverage_loss

    disagreement_loss:
        We want the two models to predict *different* classes.
        Maximise  max(softmax_A) - max(softmax_B)  in absolute value.
        Equivalently, minimise  -|max_A - max_B|.

    coverage_loss:
        Encourage high activations (proxy for covering more neurons).
        Minimise  -mean(softmax_A + softmax_B).

    Both losses are *minimised* during gradient descent, so we negate
    the objectives we want to maximise.
    """
    prob_a = F.softmax(logits_a, dim=-1)   # (1, 10)
    prob_b = F.softmax(logits_b, dim=-1)   # (1, 10)

    # ── Disagreement: push the top predictions apart ──
    max_a = prob_a.max(dim=-1).values      # scalar
    max_b = prob_b.max(dim=-1).values
    disagreement_loss = -torch.abs(max_a - max_b).mean()

    # ── Coverage proxy: maximise mean activation ──
    coverage_loss = -(prob_a.mean() + prob_b.mean()) / 2.0

    return disagreement_loss + lambda_cov * coverage_loss


# ─────────────────────────────────────────────────────────────
# 3. Single-image perturbation (DeepXplore core loop)
# ─────────────────────────────────────────────────────────────

def perturb_until_disagree(
    model_a: nn.Module,
    model_b: nn.Module,
    seed_image: torch.Tensor,       # (1, 3, 32, 32) normalised
    device: torch.device,
    steps: int = 100,
    step_size: float = 0.01,
    lambda_cov: float = 0.5,
    epsilon: float = 0.3,           # max L-inf perturbation (in normalised space)
) -> Tuple[torch.Tensor, bool, int, int]:
    """
    Iteratively perturb `seed_image` to find a disagreement input.

    Returns:
        perturbed  – final image tensor (1, 3, 32, 32)
        found      – True if the two models disagree on the perturbed image
        pred_a     – model A's predicted class index
        pred_b     – model B's predicted class index
    """
    model_a.eval()
    model_b.eval()

    # Clone seed; we optimise this tensor
    x = seed_image.clone().to(device)
    x_orig = x.clone()

    for step in range(steps):
        x = x.detach().requires_grad_(True)

        logits_a = model_a(x)
        logits_b = model_b(x)

        loss = compute_joint_loss(logits_a, logits_b, lambda_cov)
        loss.backward()

        with torch.no_grad():
            # Gradient sign step (FGSM-style)
            grad_sign = x.grad.sign()
            x = x - step_size * grad_sign   # minus: we minimise loss

            # Project back into L-inf ball around original image
            x = torch.max(torch.min(x, x_orig + epsilon), x_orig - epsilon)

            # Clamp to valid normalised range (approx −3 to 3 for CIFAR-10 stats)
            x = x.clamp(-3.0, 3.0)

        # Check for disagreement
        with torch.no_grad():
            pred_a = model_a(x).argmax(dim=1).item()
            pred_b = model_b(x).argmax(dim=1).item()

        if pred_a != pred_b:
            return x.detach(), True, pred_a, pred_b

    # Return whatever we have even if no disagreement found
    with torch.no_grad():
        pred_a = model_a(x).argmax(dim=1).item()
        pred_b = model_b(x).argmax(dim=1).item()
    return x.detach(), False, pred_a, pred_b


# ─────────────────────────────────────────────────────────────
# 4. Main DeepXplore runner
# ─────────────────────────────────────────────────────────────

def run_deepxplore(
    model_a: nn.Module,
    model_b: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_seeds: int = 200,
    steps: int = 100,
    step_size: float = 0.01,
    lambda_cov: float = 0.5,
    epsilon: float = 0.3,
    save_dir: str = "results",
    max_visualise: int = 10,
) -> Dict:
    """
    Run DeepXplore differential testing over `max_seeds` seed images.

    Returns a summary dict with:
        disagreements   – list of (perturbed_img, pred_a, pred_b, orig_label)
        coverage_a      – final neuron coverage of model A
        coverage_b      – final neuron coverage of model B
        n_seeds         – number of seeds tried
        n_found         – number of disagreement inputs found
    """
    os.makedirs(save_dir, exist_ok=True)

    # Attach coverage trackers
    tracker_a = NeuronCoverageTracker(model_a, threshold=0.5)
    tracker_b = NeuronCoverageTracker(model_b, threshold=0.5)

    disagreements = []
    n_seeds = 0

    print(f"\nRunning DeepXplore  (max_seeds={max_seeds}, steps={steps})")
    print(f"{'Seed':>6}  {'Found':>6}  {'Coverage A':>12}  {'Coverage B':>12}")
    print("-" * 45)

    for images, labels in test_loader:
        if n_seeds >= max_seeds:
            break

        for i in range(images.size(0)):
            if n_seeds >= max_seeds:
                break

            seed = images[i:i+1].to(device)
            orig_label = labels[i].item()

            # Warm up coverage trackers with the seed image
            with torch.no_grad():
                model_a(seed)
                model_b(seed)

            # Perturb
            perturbed, found, pred_a, pred_b = perturb_until_disagree(
                model_a, model_b, seed, device,
                steps=steps, step_size=step_size,
                lambda_cov=lambda_cov, epsilon=epsilon,
            )

            # Update coverage with perturbed image
            with torch.no_grad():
                model_a(perturbed)
                model_b(perturbed)

            n_seeds += 1

            if found:
                disagreements.append((
                    perturbed.cpu(), pred_a, pred_b, orig_label, seed.cpu()
                ))

            if n_seeds % 20 == 0 or found:
                cov_a = tracker_a.coverage()
                cov_b = tracker_b.coverage()
                status = "✓ DISAGREE" if found else ""
                print(f"{n_seeds:>6}  {len(disagreements):>6}  "
                      f"{cov_a:>12.4f}  {cov_b:>12.4f}  {status}")

    cov_a = tracker_a.coverage()
    cov_b = tracker_b.coverage()

    tracker_a.remove_hooks()
    tracker_b.remove_hooks()

    print(f"\n{'='*45}")
    print(f"Seeds tried        : {n_seeds}")
    print(f"Disagreements found: {len(disagreements)}")
    print(f"Final coverage A   : {cov_a:.4f} ({cov_a*100:.2f}%)")
    print(f"Final coverage B   : {cov_b:.4f} ({cov_b*100:.2f}%)")
    print(f"{'='*45}\n")

    # Visualise up to max_visualise disagreement inputs
    if disagreements:
        visualise_disagreements(disagreements[:max_visualise], save_dir)

    return {
        "disagreements": disagreements,
        "coverage_a": cov_a,
        "coverage_b": cov_b,
        "n_seeds": n_seeds,
        "n_found": len(disagreements),
    }


# ─────────────────────────────────────────────────────────────
# 5. Visualisation
# ─────────────────────────────────────────────────────────────

# CIFAR-10 denormalisation constants
_MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
_STD  = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)


def denormalise(tensor: torch.Tensor) -> np.ndarray:
    """Convert normalised CHW tensor to HWC uint8 numpy array for plotting."""
    img = tensor.squeeze(0).cpu() * _STD + _MEAN
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return (img * 255).astype(np.uint8)


def visualise_disagreements(
    disagreements: list,
    save_dir: str,
):
    """
    For each disagreement, save a side-by-side figure:
        original seed | perturbed input | difference (×10)
    """
    for idx, (perturbed, pred_a, pred_b, orig_label, seed) in enumerate(disagreements):
        orig_img  = denormalise(seed)
        pert_img  = denormalise(perturbed)

        # Amplify difference for visibility
        diff = (perturbed - seed).squeeze(0).cpu()
        diff_vis = (diff * 10 + 0.5).clamp(0, 1).permute(1, 2, 0).numpy()

        fig, axes = plt.subplots(1, 3, figsize=(9, 3))
        fig.suptitle(
            f"Disagreement #{idx+1}  |  "
            f"True: {CIFAR10_CLASSES[orig_label]}  |  "
            f"Model A: {CIFAR10_CLASSES[pred_a]}  |  "
            f"Model B: {CIFAR10_CLASSES[pred_b]}",
            fontsize=10,
        )

        axes[0].imshow(orig_img);      axes[0].set_title("Original seed")
        axes[1].imshow(pert_img);      axes[1].set_title("Perturbed input")
        axes[2].imshow(diff_vis);      axes[2].set_title("Difference (×10)")

        for ax in axes:
            ax.axis("off")

        plt.tight_layout()
        path = os.path.join(save_dir, f"disagreement_{idx+1:02d}.png")
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved → {path}")

    print(f"\nAll visualisations saved to ./{save_dir}/")


# ─────────────────────────────────────────────────────────────
# 6. Model loader utility
# ─────────────────────────────────────────────────────────────

def load_model(ckpt_path: str, device: torch.device) -> nn.Module:
    """
    Load a ResNet50-CIFAR10 checkpoint saved by train.py.
    Imports build_resnet50_cifar10 from train.py to ensure
    the architecture is identical.
    """
    from train import build_resnet50_cifar10

    model = build_resnet50_cifar10().to(device)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    val_acc = ckpt.get("val_acc", "N/A")
    print(f"Loaded {ckpt_path}  (val_acc={val_acc:.2f}%)")
    return model