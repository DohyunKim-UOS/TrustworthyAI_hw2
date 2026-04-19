"""
test.py
Demonstrates running DeepXplore on two ResNet50 models trained on CIFAR-10.

Usage:
    python test.py
    python test.py --model_a models/model_a_best.pth \
                   --model_b models/model_b_best.pth \
                   --max_seeds 200 --steps 100

Outputs:
    results/disagreement_XX.png  – visualisations of disagreement inputs
    results/summary.txt          – text summary of the run
"""

import os
import argparse
import torch
import torchvision
import torchvision.transforms as transforms

from deepxplore import run_deepxplore, load_model


# ─────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run DeepXplore differential testing on CIFAR-10 ResNet50 models"
    )
    parser.add_argument("--model_a",    default="models/model_a_best.pth",
                        help="Path to Model A checkpoint (default: models/model_a_best.pth)")
    parser.add_argument("--model_b",    default="models/model_b_best.pth",
                        help="Path to Model B checkpoint (default: models/model_b_best.pth)")
    parser.add_argument("--max_seeds",  type=int,   default=200,
                        help="Number of seed images to test (default: 200)")
    parser.add_argument("--steps",      type=int,   default=100,
                        help="Max perturbation steps per seed (default: 100)")
    parser.add_argument("--step_size",  type=float, default=0.01,
                        help="Gradient step size (default: 0.01)")
    parser.add_argument("--epsilon",    type=float, default=0.3,
                        help="Max L-inf perturbation bound (default: 0.3)")
    parser.add_argument("--lambda_cov", type=float, default=0.5,
                        help="Coverage loss weight (default: 0.5)")
    parser.add_argument("--batch_size", type=int,   default=64,
                        help="DataLoader batch size (default: 64)")
    parser.add_argument("--save_dir",   default="results",
                        help="Directory to save visualisations (default: results)")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────
# Data loader (test split only — no augmentation)
# ─────────────────────────────────────────────────────────────

def get_test_loader(batch_size: int) -> torch.utils.data.DataLoader:
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2023, 0.1994, 0.2010)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    test_set = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    return torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True
    )


# ─────────────────────────────────────────────────────────────
# Summary writer
# ─────────────────────────────────────────────────────────────

def save_summary(results: dict, args: argparse.Namespace, save_dir: str):
    """Write a plain-text summary of the DeepXplore run to results/summary.txt."""
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "summary.txt")

    lines = [
        "=" * 55,
        "DeepXplore Differential Testing — Summary",
        "=" * 55,
        "",
        "[Configuration]",
        f"  Model A checkpoint : {args.model_a}",
        f"  Model B checkpoint : {args.model_b}",
        f"  Seeds tested       : {results['n_seeds']}",
        f"  Perturbation steps : {args.steps}",
        f"  Step size          : {args.step_size}",
        f"  Epsilon (L-inf)    : {args.epsilon}",
        f"  Lambda coverage    : {args.lambda_cov}",
        "",
        "[Results]",
        f"  Disagreements found: {results['n_found']}",
        f"  Disagreement rate  : {results['n_found'] / max(results['n_seeds'], 1) * 100:.2f}%",
        f"  Neuron coverage A  : {results['coverage_a'] * 100:.2f}%",
        f"  Neuron coverage B  : {results['coverage_b'] * 100:.2f}%",
        "",
        "[Disagreement Details]",
    ]

    from deepxplore import CIFAR10_CLASSES
    for idx, (_, pred_a, pred_b, orig_label, _) in enumerate(results["disagreements"]):
        lines.append(
            f"  #{idx+1:02d}  true={CIFAR10_CLASSES[orig_label]:<12}"
            f"  model_a={CIFAR10_CLASSES[pred_a]:<12}"
            f"  model_b={CIFAR10_CLASSES[pred_b]:<12}"
        )

    lines += ["", "=" * 55]

    with open(path, "w") as f:
        f.write("\n".join(lines))

    print(f"\nSummary saved → {path}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Device ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Sanity check: model checkpoints exist ──
    for path in [args.model_a, args.model_b]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Checkpoint not found: {path}\n"
                "Please run  python train.py  first to generate the checkpoints."
            )

    # ── Load models ──
    print("\n[1/3] Loading models...")
    model_a = load_model(args.model_a, device)
    model_b = load_model(args.model_b, device)

    # ── Load CIFAR-10 test set ──
    print("\n[2/3] Loading CIFAR-10 test set...")
    test_loader = get_test_loader(args.batch_size)
    print(f"  Test set size: {len(test_loader.dataset)} images")

    # ── Run DeepXplore ──
    print("\n[3/3] Running DeepXplore...")
    results = run_deepxplore(
        model_a       = model_a,
        model_b       = model_b,
        test_loader   = test_loader,
        device        = device,
        max_seeds     = args.max_seeds,
        steps         = args.steps,
        step_size     = args.step_size,
        lambda_cov    = args.lambda_cov,
        epsilon       = args.epsilon,
        save_dir      = args.save_dir,
        max_visualise = 10,   # save up to 10 PNG visualisations
    )

    # ── Save summary ──
    save_summary(results, args, args.save_dir)

    # ── Final print ──
    print("\n✓ DeepXplore complete.")
    print(f"  Disagreements : {results['n_found']} / {results['n_seeds']} seeds")
    print(f"  Coverage A    : {results['coverage_a']*100:.2f}%")
    print(f"  Coverage B    : {results['coverage_b']*100:.2f}%")
    print(f"  Outputs saved : ./{args.save_dir}/")


if __name__ == "__main__":
    main()