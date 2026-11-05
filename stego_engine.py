"""
StegIT: Medical Image Non-ROI Steganography Engine
===================================================
Protects clinical diagnostic Region of Interest (Lungs) via U-Net segmentation
and embeds sensitive clinical JSON data strictly into Non-ROI (RONI) tissues & background.

Steps implemented:
  Step 2: Lung ROI Segmentation & Non-ROI Isolation (with morphological dilation buffer)
  Step 3: Embeddable Pixel Identification & Key-Based Permutation
  Step 4: Data Framing & Adaptive LSB Embedding in Non-ROI
  Step 5: Rigorous Verification (Strict Zero ROI Difference, PSNR, SSIM, Visual Heatmaps)
  Step 6: Bit Extraction, CRC32 Integrity Check & JSON Reconstruction
"""

import os
import cv2
import json
import zlib
import struct
import random
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from train_unet import UNet


MAGIC_HEADER = b"MSTEG"  # 5 bytes magic identifier
HEADER_FORMAT = ">5sBII" # Magic (5s), Flags (1B), PayloadLength (4B), CRC32 (4B)
HEADER_SIZE_BYTES = struct.calcsize(HEADER_FORMAT) # 14 bytes = 112 bits


# =====================================================================
# STEP 2: LUNG ROI SEGMENTATION & NON-ROI ISOLATION
# =====================================================================
def load_segmentation_model(model_path="models/unet_lung_seg_best.pth", device=None):
    """Loads the trained U-Net lung segmentation model."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(model_path):
        fallback = "models/unet_lung_seg.pth"
        if os.path.exists(fallback):
            model_path = fallback
        else:
            raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")

    model = UNet().to(device)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model, device


def segment_lung_roi(image, model, device, safety_buffer_px=7):
    """
    Runs U-Net segmentation on chest X-Ray and extracts ROI (lungs) and Non-ROI (RONI).
    Applies morphological dilation to ensure a protective safety buffer around the lungs.
    
    Returns:
      roi_mask: Binary mask where 1 = Protected Lung ROI + Buffer
      non_roi_mask: Binary mask where 1 = Safe Embeddable Zone
    """
    orig_h, orig_w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()

    # Preprocess for U-Net
    resized = cv2.resize(gray, (512, 512))
    norm_img = resized.astype(np.float32) / 255.0
    tensor = torch.tensor(norm_img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
            logit = model(tensor)
            prob = torch.sigmoid(logit).squeeze().cpu().numpy()
            pred_512 = (prob > 0.5).astype(np.uint8)

    # Scale back to native image dimensions
    raw_roi_mask = cv2.resize(pred_512, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    # Apply morphological dilation as a safety buffer zone around lungs
    if safety_buffer_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (safety_buffer_px, safety_buffer_px))
        buffered_roi_mask = cv2.dilate(raw_roi_mask, kernel, iterations=1)
    else:
        buffered_roi_mask = raw_roi_mask

    # Non-ROI (RONI) is the exact inverse
    non_roi_mask = (buffered_roi_mask == 0).astype(np.uint8)

    return buffered_roi_mask, non_roi_mask, raw_roi_mask


# =====================================================================
# STEP 3: IDENTIFY & COORDINATE EMBEDDABLE PIXELS
# =====================================================================
def get_embeddable_coordinates(non_roi_mask, key=42):
    """
    Finds all pixels in the Non-ROI mask and returns a deterministic,
    pseudo-randomly shuffled list of coordinates governed by the secret stego key.
    """
    coords = np.argwhere(non_roi_mask == 1) # Array of (row, col)
    total_pixels = len(coords)

    if total_pixels == 0:
        raise ValueError("No non-ROI pixels found! Cannot perform embedding.")

    # Reproducible pseudo-random permutation based on the key
    rng = np.random.default_rng(key)
    perm = rng.permutation(total_pixels)
    shuffled_coords = coords[perm]

    return shuffled_coords, total_pixels


# =====================================================================
# STEP 4: DATA FRAMING & LSB EMBEDDING IN NON-ROI
# =====================================================================
def prepare_bitstream(payload_bytes, compress=True):
    """
    Formats the binary payload with protocol framing:
    [ MAGIC (5B) | FLAGS (1B) | PAYLOAD_LEN (4B) | CRC32 (4B) | PAYLOAD_DATA (NB) ]
    """
    flags = 0
    if compress:
        compressed_payload = zlib.compress(payload_bytes, level=9)
        # Use compressed if smaller
        if len(compressed_payload) < len(payload_bytes):
            payload_bytes = compressed_payload
            flags |= 0x01 # Bit 0 indicates zlib compression

    payload_len = len(payload_bytes)
    payload_crc = zlib.crc32(payload_bytes) & 0xFFFFFFFF

    header = struct.pack(HEADER_FORMAT, MAGIC_HEADER, flags, payload_len, payload_crc)
    full_packet = header + payload_bytes

    # Convert packet to stream of bits (MSB first per byte)
    bits = []
    for b in full_packet:
        for i in range(7, -1, -1):
            bits.append((b >> i) & 1)

    return bits, payload_len, flags


def embed_data_non_roi(image, non_roi_mask, payload_bytes, key=42, compress=True):
    """
    Embeds the payload bits strictly into the LSB of Non-ROI pixels.
    ROI pixels are guaranteed 100% untouched.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
    stego_image = gray.copy()

    bits, original_len, flags = prepare_bitstream(payload_bytes, compress=compress)
    required_bits = len(bits)

    shuffled_coords, available_pixels = get_embeddable_coordinates(non_roi_mask, key=key)

    if required_bits > available_pixels:
        raise ValueError(
            f"Payload too large! Required: {required_bits} bits ({required_bits // 8} bytes), "
            f"Available Non-ROI Capacity: {available_pixels} bits ({available_pixels // 8} bytes)."
        )

    # Embed bits into the LSB of selected non-ROI coordinates
    for idx, bit in enumerate(bits):
        r, c = shuffled_coords[idx]
        stego_image[r, c] = (stego_image[r, c] & 0xFE) | bit

    return stego_image, required_bits, available_pixels


# =====================================================================
# STEP 5: VERIFICATION & QUALITY METRICS
# =====================================================================
def calculate_psnr(orig, stego):
    """Calculates Peak Signal-to-Noise Ratio (PSNR) in dB."""
    mse = np.mean((orig.astype(np.float64) - stego.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return 10.0 * np.log10((255.0 ** 2) / mse)


def calculate_ssim(img1, img2):
    """Computes Structural Similarity Index (SSIM) between two grayscale images."""
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(img1 ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return np.mean(ssim_map)


def verify_stego_integrity(orig_image, stego_image, roi_mask):
    """
    Verifies that the protected ROI has strictly ZERO modified pixels,
    and computes PSNR and SSIM over the image.
    """
    roi_indices = (roi_mask > 0)
    roi_orig = orig_image[roi_indices].astype(int)
    roi_stego = stego_image[roi_indices].astype(int)

    roi_diff = np.abs(roi_stego - roi_orig)
    num_roi_corrupted = int(np.count_nonzero(roi_diff))
    max_roi_diff = int(np.max(roi_diff)) if len(roi_diff) > 0 else 0

    psnr_full = calculate_psnr(orig_image, stego_image)
    ssim_full = calculate_ssim(orig_image, stego_image)

    return {
        "num_roi_corrupted_pixels": num_roi_corrupted,
        "max_roi_pixel_difference": max_roi_diff,
        "psnr_full_db": psnr_full,
        "ssim_full": ssim_full,
        "is_roi_intact": (num_roi_corrupted == 0 and max_roi_diff == 0)
    }


def generate_diagnostic_plots(orig, stego, roi_mask, raw_mask, output_path="outputs/stego_diagnostic_report.png"):
    """
    Generates a visual 4-panel diagnostic figure confirming:
    1. Original CXR
    2. Lung ROI Mask + Safety Buffer
    3. Stego Image
    4. Amplified Difference Map (x50) showing strict zero change in ROI
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    diff = cv2.absdiff(stego, orig)
    diff_amplified = diff * 50  # Amplify steganographic 1-bit noise for human visibility

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(orig, cmap='gray')
    axes[0].set_title("1. Original Chest X-Ray", fontsize=11, fontweight='bold')
    axes[0].axis("off")

    # Overlay showing raw lungs (blue) + safety buffer (green)
    mask_viz = np.zeros((*roi_mask.shape, 3), dtype=np.uint8)
    mask_viz[roi_mask > 0] = [0, 180, 0]    # Green buffer
    mask_viz[raw_mask > 0] = [0, 100, 255]   # Blue lungs
    axes[1].imshow(orig, cmap='gray')
    axes[1].imshow(mask_viz, alpha=0.45)
    axes[1].set_title("2. Protected ROI (Lungs + Buffer)", fontsize=11, fontweight='bold')
    axes[1].axis("off")

    axes[2].imshow(stego, cmap='gray')
    axes[2].set_title("3. Stego CXR (Data Embedded)", fontsize=11, fontweight='bold')
    axes[2].axis("off")

    im = axes[3].imshow(diff_amplified, cmap='inferno')
    axes[3].set_title("4. Amplified Diff (|Stego - Orig| x 50)\n(Lungs = Absolute Zero Diff)", fontsize=11, fontweight='bold')
    axes[3].axis("off")
    fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[*] Visual diagnostic report saved: '{output_path}'")


# =====================================================================
# STEP 6: BIT EXTRACTION & INTEGRITY CHECK
# =====================================================================
def extract_data_non_roi(stego_image, non_roi_mask, key=42):
    """
    Extracts embedded payload bits from the Non-ROI pixels of the stego image.
    Validates magic signature, payload length, and CRC32 integrity checksum.
    
    Returns:
      recovered_bytes: The exact original payload bytes.
    """
    gray = cv2.cvtColor(stego_image, cv2.COLOR_BGR2GRAY) if len(stego_image.shape) == 3 else stego_image

    shuffled_coords, available_pixels = get_embeddable_coordinates(non_roi_mask, key=key)

    header_bits_needed = HEADER_SIZE_BYTES * 8
    if available_pixels < header_bits_needed:
        raise ValueError("Stego image does not contain enough Non-ROI pixels to hold header.")

    # 1. Read header bits
    header_bytes_list = []
    bit_cursor = 0

    for b_idx in range(HEADER_SIZE_BYTES):
        byte_val = 0
        for shift in range(7, -1, -1):
            r, c = shuffled_coords[bit_cursor]
            bit = gray[r, c] & 1
            byte_val |= (bit << shift)
            bit_cursor += 1
        header_bytes_list.append(byte_val)

    header_bytes = bytes(header_bytes_list)

    # 2. Unpack header
    magic, flags, payload_len, expected_crc = struct.unpack(HEADER_FORMAT, header_bytes)

    if magic != MAGIC_HEADER:
        raise ValueError(
            f"Invalid steganography signature '{magic}'. "
            f"Either this image has no embedded data or the provided Stego Key ({key}) is incorrect!"
        )

    # Check capacity for full payload
    total_bits_needed = (HEADER_SIZE_BYTES + payload_len) * 8
    if total_bits_needed > available_pixels:
        raise ValueError(f"Corrupted header: encoded payload length ({payload_len} bytes) exceeds image capacity.")

    # 3. Read payload bytes
    payload_bytes_list = []
    for _ in range(payload_len):
        byte_val = 0
        for shift in range(7, -1, -1):
            r, c = shuffled_coords[bit_cursor]
            bit = gray[r, c] & 1
            byte_val |= (bit << shift)
            bit_cursor += 1
        payload_bytes_list.append(byte_val)

    raw_payload = bytes(payload_bytes_list)

    # 4. CRC32 Checksum Validation
    calculated_crc = zlib.crc32(raw_payload) & 0xFFFFFFFF
    if calculated_crc != expected_crc:
        raise ValueError(f"CRC32 Checksum Mismatch! Expected {expected_crc:#x}, calculated {calculated_crc:#x}.")

    # 5. Decompress if compressed
    is_compressed = bool(flags & 0x01)
    if is_compressed:
        recovered_bytes = zlib.decompress(raw_payload)
    else:
        recovered_bytes = raw_payload

    return recovered_bytes


# =====================================================================
# END-TO-END DEMONSTRATION & CLI
# =====================================================================
def run_end_to_end_demo(
    image_path="data/chest-xray-masks-and-labels/Lung Segmentation/CXR_png/CHNCXR_0001_0.png",
    json_path="data/dummy_clinical_data.json",
    key=42,
    safety_buffer=7
):
    """Executes complete Step 2 -> Step 6 pipeline test."""
    print("=================================================================")
    print("           STEGIT NON-ROI STEGANOGRAPHY PIPELINE DEMO           ")
    print("=================================================================")

    # 0. Load files
    print(f"[*] Loading input X-Ray: '{image_path}'")
    orig_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if orig_image is None:
        raise FileNotFoundError(f"Image not found at: {image_path}")

    print(f"[*] Reading dummy clinical JSON: '{json_path}'")
    with open(json_path, "r", encoding="utf-8") as f:
        json_content = f.read()
    json_bytes = json_content.encode("utf-8")
    print(f"    Payload size: {len(json_bytes)} bytes ({len(json_bytes)*8} bits)")

    # Step 2: Segmentation
    print("\n--- [Step 2] U-Net Lung ROI Segmentation ---")
    model, device = load_segmentation_model()
    roi_mask, non_roi_mask, raw_mask = segment_lung_roi(
        orig_image, model, device, safety_buffer_px=safety_buffer
    )
    lung_pixels = np.count_nonzero(raw_mask)
    buffer_pixels = np.count_nonzero(roi_mask) - lung_pixels
    non_roi_pixels = np.count_nonzero(non_roi_mask)
    total_px = orig_image.size
    print(f"    Total Pixels:    {total_px:,}")
    print(f"    Lung ROI:        {lung_pixels:,} px ({lung_pixels/total_px*100:.1f}%) [STRICTLY PROTECTED]")
    print(f"    Safety Buffer:   {buffer_pixels:,} px ({buffer_pixels/total_px*100:.1f}%) [PROTECTED]")
    print(f"    Safe Non-ROI:    {non_roi_pixels:,} px ({non_roi_pixels/total_px*100:.1f}%) [EMBEDDING ZONE]")

    # Save companion mask
    os.makedirs("outputs", exist_ok=True)
    mask_path = "outputs/lung_roi_mask.png"
    cv2.imwrite(mask_path, (roi_mask * 255).astype(np.uint8))
    print(f"    Companion ROI Mask saved to: '{mask_path}'")

    # Step 3 & 4: Embedding
    print("\n--- [Step 3 & 4] Non-ROI Embedding ---")
    stego_image, bits_embedded, capacity = embed_data_non_roi(
        orig_image, non_roi_mask, json_bytes, key=key, compress=True
    )
    print(f"    Stego Key:       {key}")
    print(f"    Bits Embedded:   {bits_embedded:,} bits ({bits_embedded // 8} bytes incl. headers)")
    print(f"    Capacity Used:   {(bits_embedded / capacity)*100:.2f}% of Non-ROI")
    
    stego_path = "outputs/stego_chest_xray.png"
    cv2.imwrite(stego_path, stego_image)
    print(f"    Stego CXR Image saved: '{stego_path}'")

    # Step 5: Verification
    print("\n--- [Step 5] Integrity & Quality Verification ---")
    metrics = verify_stego_integrity(orig_image, stego_image, roi_mask)
    print(f"    ROI Corrupted Pixels:    {metrics['num_roi_corrupted_pixels']} (MUST BE 0)")
    print(f"    Max ROI Pixel Diff:      {metrics['max_roi_pixel_difference']} (MUST BE 0)")
    print(f"    Full Image PSNR:         {metrics['psnr_full_db']:.2f} dB (> 50 dB is exceptional)")
    print(f"    Full Image SSIM:         {metrics['ssim_full']:.6f} (> 0.999 is ideal)")
    print(f"    Strict ROI Preservation: {' PASS (100% UNTOUCHED)' if metrics['is_roi_intact'] else ' FAIL'}")

    generate_diagnostic_plots(orig_image, stego_image, roi_mask, raw_mask, output_path="outputs/stego_diagnostic_report.png")

    # Step 6: Extraction
    print("\n--- [Step 6] Extraction & Reconstruction ---")
    recovered_bytes = extract_data_non_roi(stego_image, non_roi_mask, key=key)
    recovered_json_str = recovered_bytes.decode("utf-8")
    recovered_data = json.loads(recovered_json_str)

    extracted_json_path = "outputs/extracted_clinical_data.json"
    with open(extracted_json_path, "w", encoding="utf-8") as f:
        json.dump(recovered_data, f, indent=2)

    # Validate byte-exact equality
    is_identical = (recovered_bytes == json_bytes)
    print(f"    Extracted JSON Saved:    '{extracted_json_path}'")
    print(f"    Patient ID in Extracted: {recovered_data.get('patient_profile', {}).get('patient_id')}")
    print(f"    Radiology Impression:    {recovered_data.get('radiology_study', {}).get('radiologist_impression')[:60]}...")
    print(f"    Byte-Exact Integrity:    {' PERFECT MATCH (100.0%)' if is_identical else ' MISMATCH'}")
    print("=================================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="StegIT Non-ROI Steganography Engine")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Demo
    demo_parser = subparsers.add_parser("demo", help="Run full end-to-end embedding and extraction demo")
    demo_parser.add_argument("--image", default="data/chest-xray-masks-and-labels/Lung Segmentation/CXR_png/CHNCXR_0001_0.png")
    demo_parser.add_argument("--json", default="data/dummy_clinical_data.json")
    demo_parser.add_argument("--key", type=int, default=42)

    # Embed
    embed_parser = subparsers.add_parser("embed", help="Embed JSON data into CXR Non-ROI")
    embed_parser.add_argument("--image", required=True, help="Input Chest X-Ray image")
    embed_parser.add_argument("--data", required=True, help="Input JSON file with clinical data")
    embed_parser.add_argument("--output", default="outputs/stego_chest_xray.png", help="Output stego image path")
    embed_parser.add_argument("--key", type=int, default=42, help="Stego secret key")
    embed_parser.add_argument("--buffer", type=int, default=7, help="Safety buffer dilation kernel size in pixels")

def check_stego_presence(image_or_path, mask_or_path=None, key=42):
    """
    Inspects an image to check if valid steganographic data is embedded.
    Reads only the protocol header (14 bytes) to test for the 'MSTEG' magic signature.
    """
    if isinstance(image_or_path, str):
        img = cv2.imread(image_or_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Image not found at {image_or_path}")
    else:
        img = image_or_path

    if mask_or_path is not None:
        if isinstance(mask_or_path, str):
            mask_img = cv2.imread(mask_or_path, cv2.IMREAD_GRAYSCALE)
        else:
            mask_img = mask_or_path
        non_roi = (mask_img == 0).astype(np.uint8)
    else:
        # Automatically generate mask using U-Net
        model, dev = load_segmentation_model()
        _, non_roi, _ = segment_lung_roi(img, model, dev)

    shuffled_coords, available_pixels = get_embeddable_coordinates(non_roi, key=key)
    header_bits_needed = HEADER_SIZE_BYTES * 8

    if available_pixels < header_bits_needed:
        return {"has_data": False, "reason": "Insufficient non-ROI pixels"}

    header_bytes_list = []
    bit_cursor = 0
    for _ in range(HEADER_SIZE_BYTES):
        byte_val = 0
        for shift in range(7, -1, -1):
            r, c = shuffled_coords[bit_cursor]
            bit = img[r, c] & 1
            byte_val |= (bit << shift)
            bit_cursor += 1
        header_bytes_list.append(byte_val)

    header_bytes = bytes(header_bytes_list)
    magic, flags, payload_len, crc32 = struct.unpack(HEADER_FORMAT, header_bytes)

    if magic == MAGIC_HEADER:
        return {
            "has_data": True,
            "magic": magic.decode("ascii", errors="ignore"),
            "compressed": bool(flags & 0x01),
            "payload_len_bytes": payload_len,
            "crc32": f"{crc32:#010x}",
            "key": key
        }
    else:
        return {
            "has_data": False,
            "magic_found": str(magic),
            "reason": "Magic signature mismatch (Clean image or incorrect key)"
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="StegIT Non-ROI Steganography Engine")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Demo
    demo_parser = subparsers.add_parser("demo", help="Run full end-to-end embedding and extraction demo")
    demo_parser.add_argument("--image", default="data/chest-xray-masks-and-labels/Lung Segmentation/CXR_png/CHNCXR_0001_0.png")
    demo_parser.add_argument("--json", default="data/dummy_clinical_data.json")
    demo_parser.add_argument("--key", type=int, default=42)

    # Embed
    embed_parser = subparsers.add_parser("embed", help="Embed JSON data into CXR Non-ROI")
    embed_parser.add_argument("--image", required=True, help="Input Chest X-Ray image")
    embed_parser.add_argument("--data", required=True, help="Input JSON file with clinical data")
    embed_parser.add_argument("--output", default="outputs/stego_chest_xray.png", help="Output stego image path")
    embed_parser.add_argument("--key", type=int, default=42, help="Stego secret key")
    embed_parser.add_argument("--buffer", type=int, default=7, help="Safety buffer dilation kernel size in pixels")

    # Extract
    extract_parser = subparsers.add_parser("extract", help="Extract JSON data from Stego CXR")
    extract_parser.add_argument("--stego", required=True, help="Stego Chest X-Ray image")
    extract_parser.add_argument("--mask", required=True, help="Companion Lung ROI mask (from embed step)")
    extract_parser.add_argument("--output", default="outputs/extracted_clinical_data.json", help="Output JSON path")
    extract_parser.add_argument("--key", type=int, default=42, help="Stego secret key")

    # Check / Inspect
    check_parser = subparsers.add_parser("check", help="Check if an image contains embedded steganographic data")
    check_parser.add_argument("--image", required=True, help="Chest X-Ray image to inspect")
    check_parser.add_argument("--mask", default=None, help="Companion Lung ROI mask (optional, will compute if omitted)")
    check_parser.add_argument("--key", type=int, default=42, help="Stego secret key to check against")

    args = parser.parse_args()

    if args.command == "embed":
        img = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
        with open(args.data, "r", encoding="utf-8") as f:
            data_bytes = f.read().encode("utf-8")

        model, dev = load_segmentation_model()
        roi_mask, non_roi_mask, raw_mask = segment_lung_roi(img, model, dev, safety_buffer_px=args.buffer)

        stego, bits, cap = embed_data_non_roi(img, non_roi_mask, data_bytes, key=args.key)
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        cv2.imwrite(args.output, stego)

        mask_out = os.path.splitext(args.output)[0] + "_mask.png"
        cv2.imwrite(mask_out, (roi_mask * 255).astype(np.uint8))
        print(f"[*] Stego image saved to: {args.output}")
        print(f"[*] Companion mask saved to: {mask_out}")

    elif args.command == "extract":
        stego_img = cv2.imread(args.stego, cv2.IMREAD_GRAYSCALE)
        mask_img = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
        non_roi = (mask_img == 0).astype(np.uint8)

        recovered = extract_data_non_roi(stego_img, non_roi, key=args.key)
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "wb") as f:
            f.write(recovered)
        print(f"[*] Extracted data saved to: {args.output}")

    elif args.command == "check":
        res = check_stego_presence(args.image, args.mask, key=args.key)
        print("\n=======================================================")
        print("          STEGANOGRAPHY PRESENCE INSPECTION            ")
        print("=======================================================")
        print(f" Target Image: {args.image}")
        print(f" Stego Key:    {args.key}")
        if res["has_data"]:
            print(f" Status:       [+] EMBEDDED DATA DETECTED!")
            print(f" Signature:    '{res['magic']}' (Valid Medical Steg Header)")
            print(f" Payload Size: {res['payload_len_bytes']:,} bytes")
            print(f" Compression:  {'Enabled (zlib)' if res['compressed'] else 'Raw'}")
            print(f" CRC32:        {res['crc32']}")
        else:
            print(f" Status:       [-] NO DATA DETECTED (Clean image or invalid key)")
            print(f" Reason:       {res['reason']}")
        print("=======================================================\n")

    else:
        # Default to running the full demo
        run_end_to_end_demo()

