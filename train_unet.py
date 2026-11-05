import os
import cv2
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

# 1. Dataset Path
REPO_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(REPO_DIR, "data", "chest-xray-masks-and-labels")
BASE_PATH = os.path.join(DATASET_DIR, "Lung Segmentation")
IMAGE_DIR = os.path.join(BASE_PATH, "CXR_png")
MASK_DIR = os.path.join(BASE_PATH, "masks")


# 2. Dataset Definition with Augmentations
class LungDataset(Dataset):
    def __init__(self, image_dir, mask_dir, augment=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.augment = augment
        
        # Collect valid image files that have corresponding masks
        all_images = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.valid_pairs = []
        
        for img_name in all_images:
            mask_name1 = img_name.replace(".png", "_mask.png")
            mask_name2 = img_name
            
            p1 = os.path.join(mask_dir, mask_name1)
            p2 = os.path.join(mask_dir, mask_name2)
            
            if os.path.exists(p1):
                self.valid_pairs.append((img_name, mask_name1))
            elif os.path.exists(p2):
                self.valid_pairs.append((img_name, mask_name2))

    def __len__(self):
        return len(self.valid_pairs)

    def __getitem__(self, idx):
        img_name, mask_name = self.valid_pairs[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            return self.__getitem__((idx + 1) % len(self))

        image = cv2.resize(image, (512, 512))
        mask = cv2.resize(mask, (512, 512))

        # Data Augmentation for training
        if self.augment:
            # Random Horizontal Flip
            if random.random() > 0.5:
                image = cv2.flip(image, 1)
                mask = cv2.flip(mask, 1)
            
            # Random slight rotation (-10 to +10 degrees)
            if random.random() > 0.5:
                angle = random.uniform(-10, 10)
                M = cv2.getRotationMatrix2D((256, 256), angle, 1.0)
                image = cv2.warpAffine(image, M, (512, 512), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                mask = cv2.warpAffine(mask, M, (512, 512), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)

        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        mask = (mask > 127).astype(np.float32)  # Binarize mask cleanly

        image = np.expand_dims(image, axis=0)
        mask = np.expand_dims(mask, axis=0)

        return torch.tensor(image, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)


# 3. Model Definition
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )

        self.enc1 = conv_block(1, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = conv_block(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = conv_block(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = conv_block(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = conv_block(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = conv_block(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = conv_block(128, 64)

        self.final = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        b = self.bottleneck(self.pool4(e4))
        
        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        return self.final(d1)


# 4. Hybrid BCEDice Loss (AMP-Safe)
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(2, 3))
        dice = (2.0 * intersection + self.smooth) / (probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) + self.smooth)
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        return self.bce_weight * self.bce(logits, targets) + self.dice_weight * self.dice(logits, targets)


# 5. Metrics calculation
def calculate_metrics(logits, targets, threshold=0.5):
    probs = torch.sigmoid(logits)
    preds_bin = (probs > threshold).float()
    
    # Pixel Accuracy
    correct = (preds_bin == targets).float().sum()
    pixel_acc = correct / targets.numel()
    
    # Dice Coefficient
    intersection = (preds_bin * targets).sum()
    dice = (2.0 * intersection + 1e-7) / (preds_bin.sum() + targets.sum() + 1e-7)
    
    # IoU
    union = preds_bin.sum() + targets.sum() - intersection
    iou = (intersection + 1e-7) / (union + 1e-7)
    
    return pixel_acc.item(), dice.item(), iou.item()


# 6. Training & Validation Loop
def train(epochs=15, batch_size=2, lr=5e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== U-Net Segmentation Training ===")
    print(f"Device: {device}")
    print(f"Epochs: {epochs} | Batch Size: {batch_size} | Learning Rate: {lr}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    full_dataset = LungDataset(IMAGE_DIR, MASK_DIR, augment=True)
    val_size = max(int(0.15 * len(full_dataset)), 1)
    train_size = len(full_dataset) - val_size

    torch.manual_seed(42)
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    val_ds.dataset.augment = False # Disable augmentation on validation

    print(f"Dataset: {len(full_dataset)} total images ({train_size} train, {val_size} validation)")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=torch.cuda.is_available())

    model = UNet().to(device)
    criterion = BCEDiceLoss(bce_weight=0.4, dice_weight=0.6)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda' if torch.cuda.is_available() else 'cpu')

    best_val_dice = 0.0
    os.makedirs("models", exist_ok=True)
    save_path = os.path.join("models", "unet_lung_seg.pth")
    best_save_path = os.path.join("models", "unet_lung_seg_best.pth")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/{epochs:02d} [Train]")

        for images, masks in train_bar:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                preds = model(images)
                loss = criterion(preds, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        avg_train_loss = train_loss / len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_pixel_acc = 0.0
        val_dice = 0.0
        val_iou = 0.0

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)

                with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                    preds = model(images)
                    loss = criterion(preds, masks)

                val_loss += loss.item()
                acc, dice, iou = calculate_metrics(preds, masks)
                val_pixel_acc += acc
                val_dice += dice
                val_iou += iou

        avg_val_loss = val_loss / len(val_loader)
        avg_val_pixel_acc = (val_pixel_acc / len(val_loader)) * 100
        avg_val_dice = (val_dice / len(val_loader)) * 100
        avg_val_iou = (val_iou / len(val_loader)) * 100

        print(
            f"Epoch {epoch+1:02d}/{epochs:02d} Results:\n"
            f"  Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}\n"
            f"  Val Pixel Acc: {avg_val_pixel_acc:.2f}% | Val Dice: {avg_val_dice:.2f}% | Val IoU: {avg_val_iou:.2f}%\n"
        )

        # Save current model
        torch.save(model.state_dict(), save_path)

        # Save best model
        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            torch.save(model.state_dict(), best_save_path)
            print(f"⭐ New Best Model Saved! (Val Dice: {best_val_dice:.2f}%)\n")

    print(f"Training Complete! Best Validation Dice: {best_val_dice:.2f}%")
    print(f"Models saved at: '{save_path}' and '{best_save_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train U-Net for Lung Segmentation")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs (default: 15)")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size (default: 2)")
    parser.add_argument("--lr", type=float, default=5e-4, help="Initial learning rate (default: 5e-4)")
    args = parser.parse_args()

    train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
