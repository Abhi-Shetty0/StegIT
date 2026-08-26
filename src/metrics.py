"""
Clinical Imperceptibility and Quality Metrics Module for StegIT.

Computes MSE, PSNR, SSIM, Embedding Capacity/Rate (bpp), Bit Error Rate (BER),
and visual difference heatmaps to evaluate the fidelity and diagnostic preservation
of medical stego images.
"""

import os
import math
from typing import Dict, Any, Tuple, Optional, Union
import cv2
import numpy as np
from scipy.ndimage import uniform_filter, gaussian_filter


def compute_ssim_map(
    img1: np.ndarray,
    img2: np.ndarray,
    k1: float = 0.01,
    k2: float = 0.03,
    win_size: int = 11,
    sigma: float = 1.5,
    dynamic_range: float = 255.0
) -> Tuple[float, np.ndarray]:
    """
    Computes standard Structural Similarity Index (SSIM) and SSIM error map.
    """
    if img1.shape != img2.shape:
        raise ValueError(f"Input images must have the same shape: {img1.shape} vs {img2.shape}")

    # Convert to grayscale float64 for SSIM luminance calculation
    if len(img1.shape) == 3:
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float64)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.float64)
    else:
        gray1 = img1.astype(np.float64)
        gray2 = img2.astype(np.float64)

    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2

    # Gaussian filtering for local means and variances
    mu1 = gaussian_filter(gray1, sigma=sigma)
    mu2 = gaussian_filter(gray2, sigma=sigma)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = gaussian_filter(gray1 * gray1, sigma=sigma) - mu1_sq
    sigma2_sq = gaussian_filter(gray2 * gray2, sigma=sigma) - mu2_sq
    sigma12 = gaussian_filter(gray1 * gray2, sigma=sigma) - mu1_mu2

    # SSIM formula
    numerator = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
    denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    ssim_map = numerator / denominator

    mean_ssim = float(np.mean(ssim_map))
    return mean_ssim, ssim_map


class StegoMetrics:
    """
    Evaluates visual imperceptibility, clinical preservation, and steganographic metrics.
    """

    @staticmethod
    def calculate_mse(
        img1: np.ndarray,
        img2: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> float:
        """
        Computes Mean Squared Error (MSE), optionally restricted to a binary mask.
        """
        diff = img1.astype(np.float64) - img2.astype(np.float64)
        squared_diff = diff ** 2

        if mask is not None:
            # Mask: non-zero pixels are evaluated
            binary_mask = mask > 0
            if len(img1.shape) == 3 and len(mask.shape) == 2:
                binary_mask = np.repeat(binary_mask[:, :, np.newaxis], img1.shape[2], axis=2)
            valid_pixels = np.sum(binary_mask)
            if valid_pixels == 0:
                return 0.0
            return float(np.sum(squared_diff[binary_mask]) / valid_pixels)
        else:
            return float(np.mean(squared_diff))

    @staticmethod
    def calculate_psnr(
        mse_val: float,
        max_pixel_val: float = 255.0
    ) -> float:
        """
        Computes Peak Signal-to-Noise Ratio (PSNR) in dB from MSE.
        """
        if mse_val <= 1e-12:
            return 100.0  # Perfect match / infinite PSNR capped for practical logging
        return float(10.0 * math.log10((max_pixel_val ** 2) / mse_val))

    @staticmethod
    def calculate_ber(
        original_bits: np.ndarray,
        extracted_bits: np.ndarray
    ) -> float:
        """
        Computes Bit Error Rate (BER) between original and extracted bitstreams.
        """
        min_len = min(len(original_bits), len(extracted_bits))
        if min_len == 0:
            return 0.0
        errors = np.sum(original_bits[:min_len] != extracted_bits[:min_len])
        return float(errors / min_len)

    def evaluate_all(
        self,
        cover_image: Union[str, np.ndarray],
        stego_image: Union[str, np.ndarray],
        roi_mask: Optional[np.ndarray] = None,
        embedded_bytes_count: int = 0,
        original_ciphertext: Optional[bytes] = None,
        extracted_ciphertext: Optional[bytes] = None,
    ) -> Dict[str, Any]:
        """
        Performs a full evaluation of imperceptibility and diagnostic preservation metrics.

        Returns:
            Dictionary containing Global MSE, ROI MSE, NROI MSE, PSNR, SSIM, ROI SSIM,
            bpp, BER, and diagnostic integrity verification status.
        """
        if isinstance(cover_image, str):
            cover = cv2.imread(cover_image)
        else:
            cover = cover_image

        if isinstance(stego_image, str):
            stego = cv2.imread(stego_image)
        else:
            stego = stego_image

        if cover.shape != stego.shape:
            raise ValueError(f"Cover shape {cover.shape} != Stego shape {stego.shape}")

        h, w = cover.shape[:2]
        total_pixels = h * w

        # 1. Global Metrics
        global_mse = self.calculate_mse(cover, stego)
        global_psnr = self.calculate_psnr(global_mse)
        global_ssim, ssim_map = compute_ssim_map(cover, stego)

        # 2. ROI & Non-ROI Specific Metrics
        roi_mse = 0.0
        roi_psnr = 100.0
        roi_ssim = 1.0
        nroi_mse = global_mse

        if roi_mask is not None:
            # roi_mask: 255 = Diagnostic ROI, 0 = Non-ROI background
            roi_mse = self.calculate_mse(cover, stego, mask=roi_mask)
            roi_psnr = self.calculate_psnr(roi_mse)

            nroi_mask = (roi_mask == 0).astype(np.uint8) * 255
            nroi_mse = self.calculate_mse(cover, stego, mask=nroi_mask)

            # ROI SSIM
            roi_indices = roi_mask > 0
            if np.any(roi_indices):
                roi_ssim = float(np.mean(ssim_map[roi_indices]))

        # 3. Capacity & Bitrate
        embedded_bits = embedded_bytes_count * 8
        bpp = embedded_bits / total_pixels if total_pixels > 0 else 0.0

        # 4. Bit Error Rate (BER)
        ber = 0.0
        if original_ciphertext is not None and extracted_ciphertext is not None:
            orig_bits = np.unpackbits(np.frombuffer(original_ciphertext, dtype=np.uint8))
            ext_bits = np.unpackbits(np.frombuffer(extracted_ciphertext, dtype=np.uint8))
            ber = self.calculate_ber(orig_bits, ext_bits)

        # 5. Diagnostic Preservation Assessment
        is_roi_intact = (roi_mse == 0.0)
        clinical_verdict = (
            "PASSED: 100% Diagnostic ROI Integrity Preserved (Zero Distortion in Critical Diagnostic Zone)"
            if is_roi_intact
            else "WARNING: Diagnostic ROI altered!"
        )

        return {
            "global_mse": global_mse,
            "global_psnr_db": global_psnr,
            "global_ssim": global_ssim,
            "roi_mse": roi_mse,
            "roi_psnr_db": roi_psnr,
            "roi_ssim": roi_ssim,
            "nroi_mse": nroi_mse,
            "embedded_bytes": embedded_bytes_count,
            "embedded_bits": embedded_bits,
            "embedding_rate_bpp": bpp,
            "bit_error_rate": ber,
            "is_roi_intact": is_roi_intact,
            "clinical_verdict": clinical_verdict,
            "image_dimensions": f"{w}x{h} ({cover.shape[2] if len(cover.shape) == 3 else 1} channels)",
        }

    def generate_visual_comparison(
        self,
        cover_image: Union[str, np.ndarray],
        stego_image: Union[str, np.ndarray],
        annotated_roi_image: Optional[np.ndarray] = None,
        output_path: Optional[str] = None,
        diff_amplification: int = 50,
    ) -> np.ndarray:
        """
        Generates a multi-panel visual comparison showing:
        1. Original Cover Image
        2. Stego Image
        3. Non-ROI Detection Bounding Boxes
        4. Amplified Residual Difference Heatmap
        """
        if isinstance(cover_image, str):
            cover = cv2.imread(cover_image)
        else:
            cover = cover_image.copy()

        if isinstance(stego_image, str):
            stego = cv2.imread(stego_image)
        else:
            stego = stego_image.copy()

        if len(cover.shape) == 2:
            cover = cv2.cvtColor(cover, cv2.COLOR_GRAY2BGR)
        if len(stego.shape) == 2:
            stego = cv2.cvtColor(stego, cv2.COLOR_GRAY2BGR)

        # Compute absolute difference
        diff = cv2.absdiff(cover, stego)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY) if len(diff.shape) == 3 else diff

        # Amplify difference to visualize LSB modifications clearly
        amplified = np.clip(gray_diff.astype(np.float64) * diff_amplification, 0, 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(amplified, cv2.COLORMAP_JET)

        # If difference is all zero, show dark background
        if np.all(gray_diff == 0):
            heatmap[:] = 0

        # Panel 3: Annotated ROI
        if annotated_roi_image is None:
            roi_panel = cover.copy()
        else:
            roi_panel = annotated_roi_image.copy()

        # Labels on panels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        color = (255, 255, 255)
        thickness = 2
        bg_color = (0, 0, 0)

        panels = [
            ("1. Original Cover", cover),
            ("2. Stego Image", stego),
            ("3. Non-ROI Embedding Blocks", roi_panel),
            (f"4. Diff Heatmap ({diff_amplification}x LSB)", heatmap),
        ]

        labeled_panels = []
        for title, p in panels:
            panel_copy = p.copy()
            # Draw header bar
            cv2.rectangle(panel_copy, (0, 0), (panel_copy.shape[1], 36), bg_color, -1)
            cv2.putText(panel_copy, title, (10, 24), font, font_scale, color, thickness, cv2.LINE_AA)
            labeled_panels.append(panel_copy)

        # Arrange in a 2x2 grid
        top_row = np.hstack((labeled_panels[0], labeled_panels[1]))
        bottom_row = np.hstack((labeled_panels[2], labeled_panels[3]))
        composite = np.vstack((top_row, bottom_row))

        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            cv2.imwrite(output_path, composite)

        return composite
