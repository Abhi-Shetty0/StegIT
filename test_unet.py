import os
import cv2
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from tqdm import tqdm

from train_unet import UNet, LungDataset, calculate_metrics, BCEDiceLoss, IMAGE_DIR, MASK_DIR


def load_trained_model(model_path="models/unet_lung_seg_best.pth", device=None):
    """Load the trained U-Net model weights."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(model_path):
        fallback = "models/unet_lung_seg.pth"
        if os.path.exists(fallback):
            print(f"[!] Warning: '{model_path}' not found. Falling back to '{fallback}'.")
            model_path = fallback
        else:
            raise FileNotFoundError(f"Model file not found at '{model_path}' or '{fallback}'.")

    model = UNet().to(device)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded model weights from: '{model_path}' (Device: {device})")
    return model, device


def evaluate_dataset(model, device, batch_size=4, num_samples=None):
    """
    Evaluates the model over the validation set and prints quantitative metrics
    (Dice Coefficient, IoU / Jaccard Index, Pixel Accuracy, Loss).
    """
    full_dataset = LungDataset(IMAGE_DIR, MASK_DIR, augment=False)
    val_size = max(int(0.15 * len(full_dataset)), 1)
    train_size = len(full_dataset) - val_size

    torch.manual_seed(42)
    _, val_ds = random_split(full_dataset, [train_size, val_size])

    print(f"\n==========================================")
    print(f"       MODEL QUANTITATIVE EVALUATION      ")
    print(f"==========================================")
    print(f"Evaluating on {len(val_ds)} validation images...")

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    criterion = BCEDiceLoss(bce_weight=0.4, dice_weight=0.6)

    total_loss = 0.0
    total_pixel_acc = 0.0
    total_dice = 0.0
    total_iou = 0.0

    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="Evaluating"):
            images = images.to(device)
            masks = masks.to(device)

            with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                preds = model(images)
                loss = criterion(preds, masks)

            total_loss += loss.item()
            acc, dice, iou = calculate_metrics(preds, masks)
            total_pixel_acc += acc
            total_dice += dice
            total_iou += iou

    n_batches = len(val_loader)
    avg_loss = total_loss / n_batches
    avg_pixel_acc = (total_pixel_acc / n_batches) * 100
    avg_dice = (total_dice / n_batches) * 100
    avg_iou = (total_iou / n_batches) * 100

    print(f"\n--- Validation Metrics ---")
    print(f"  Test Loss:        {avg_loss:.4f}")
    print(f"  Dice Score:       {avg_dice:.2f}% (Higher is better)")
    print(f"  IoU / Jaccard:    {avg_iou:.2f}% (Higher is better)")
    print(f"  Pixel Accuracy:   {avg_pixel_acc:.2f}%\n")

    return avg_dice, avg_iou, avg_pixel_acc


def visualize_predictions(model, device, num_samples=5, output_dir="outputs"):
    """
    Picks random test samples, predicts masks, and generates visual side-by-side plots:
    [Original X-Ray | Ground Truth Mask | Predicted Mask | Overlay]
    """
    os.makedirs(output_dir, exist_ok=True)
    dataset = LungDataset(IMAGE_DIR, MASK_DIR, augment=False)
    
    if len(dataset) == 0:
        print("[!] No images found in dataset directory for visualization.")
        return

    num_samples = min(num_samples, len(dataset))
    random.seed(42)
    sample_indices = random.sample(range(len(dataset)), num_samples)

    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
    if num_samples == 1:
        axes = np.expand_dims(axes, axis=0)

    print(f"Generating visual predictions for {num_samples} sample images...")

    with torch.no_grad():
        for i, idx in enumerate(sample_indices):
            img_tensor, mask_tensor = dataset[idx]
            img_input = img_tensor.unsqueeze(0).to(device)

            with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                logit = model(img_input)
                prob = torch.sigmoid(logit).squeeze().cpu().numpy()
                pred_mask = (prob > 0.5).astype(np.uint8)

            orig_img = img_tensor.squeeze().cpu().numpy()
            gt_mask = mask_tensor.squeeze().cpu().numpy()

            # Create RGB overlay
            img_rgb = np.stack([orig_img] * 3, axis=-1)
            overlay = img_rgb.copy()
            # Green highlight for predicted lung regions
            overlay[:, :, 1] = np.where(pred_mask == 1, np.clip(overlay[:, :, 1] + 0.4, 0, 1), overlay[:, :, 1])
            # Red highlight for ground truth regions not captured
            overlay[:, :, 0] = np.where((gt_mask == 1) & (pred_mask == 0), 1.0, overlay[:, :, 0])

            # Subplots
            axes[i, 0].imshow(orig_img, cmap='gray')
            axes[i, 0].set_title(f"Sample #{idx+1} - Original X-Ray")
            axes[i, 0].axis("off")

            axes[i, 1].imshow(gt_mask, cmap='gray')
            axes[i, 1].set_title("Ground Truth Mask")
            axes[i, 1].axis("off")

            axes[i, 2].imshow(pred_mask, cmap='gray')
            axes[i, 2].set_title("Predicted Mask")
            axes[i, 2].axis("off")

            axes[i, 3].imshow(overlay)
            axes[i, 3].set_title("Overlay (Green=Pred, Red=Missed)")
            axes[i, 3].axis("off")

    plt.tight_layout()
    output_plot_path = os.path.join(output_dir, "segmentation_results.png")
    plt.savefig(output_plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f" Visual results saved to: '{output_plot_path}'")


def predict_single_image(model, device, image_path, output_path="outputs/single_prediction.png"):
    """Run segmentation on any single arbitrary image file."""
    if not os.path.exists(image_path):
        print(f"[!] File not found: {image_path}")
        return

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"[!] Could not decode image: {image_path}")
        return

    orig_h, orig_w = img.shape
    resized = cv2.resize(img, (512, 512))
    norm_img = resized.astype(np.float32) / 255.0
    img_tensor = torch.tensor(norm_img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
            logit = model(img_tensor)
            prob = torch.sigmoid(logit).squeeze().cpu().numpy()
            pred_mask_512 = (prob > 0.5).astype(np.uint8)

    # Resize mask back to original dimensions
    pred_mask_orig = cv2.resize(pred_mask_512, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title("Input X-Ray")
    axes[0].axis("off")

    axes[1].imshow(pred_mask_orig, cmap='gray')
    axes[1].set_title("Predicted Lung Mask")
    axes[1].axis("off")

    # Overlay
    norm_orig = img.astype(np.float32) / 255.0
    overlay = np.stack([norm_orig] * 3, axis=-1)
    overlay[:, :, 1] = np.where(pred_mask_orig == 1, np.clip(overlay[:, :, 1] + 0.4, 0, 1), overlay[:, :, 1])

    axes[2].imshow(overlay)
    axes[2].set_title("Lung Segmentation Overlay")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f" Single image prediction saved to: '{output_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test and Evaluate Trained U-Net Lung Segmentation Model")
    parser.add_argument("--model", type=str, default="models/unet_lung_seg_best.pth", help="Path to model checkpoint")
    parser.add_argument("--samples", type=int, default=5, help="Number of sample visualizations to generate (default: 5)")
    parser.add_argument("--image", type=str, default=None, help="Path to a single test image for inference")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory to save plots/masks")
    parser.add_argument("--no_eval", action="store_true", help="Skip full quantitative validation metrics calculation")
    args = parser.parse_args()

    # 1. Load Model
    model, device = load_trained_model(args.model)

    # 2. Single Image Mode or Dataset Mode
    if args.image:
        out_file = os.path.join(args.output_dir, "custom_inference.png")
        predict_single_image(model, device, args.image, output_path=out_file)
    else:
        # Quantitative metrics evaluation
        if not args.no_eval:
            evaluate_dataset(model, device)

        # Visual inspection side-by-side plots
        visualize_predictions(model, device, num_samples=args.samples, output_dir=args.output_dir)
