"""
Non-ROI (Region of Non-Interest) Block Detection Module for StegIT.

Evaluates medical cover images using block-wise statistical variance and mean intensity.
Flags low-detail, uniform background areas (NROI) for steganographic embedding while
strictly preserving active clinical diagnostic zones (ROI).
"""

import os
from typing import List, Tuple, Dict, Any, Union, Optional
import cv2
import numpy as np


class NonROIDetector:
    """
    Detects Non-ROI embedding blocks and dedicated header blocks in medical images.
    """

    def __init__(
        self,
        block_size: int = 48,
        var_threshold: float = 120.0,
        dark_mean_th: float = 30.0,
        dark_var_th: float = 50.0,
        dense_white_mean_th: float = 180.0,
        dense_white_var_th: float = 300.0,
    ):
        self.block_size = block_size
        self.var_threshold = var_threshold
        self.dark_mean_th = dark_mean_th
        self.dark_var_th = dark_var_th
        self.dense_white_mean_th = dense_white_mean_th
        self.dense_white_var_th = dense_white_var_th

    def detect_blocks(
        self,
        image_input: Union[str, np.ndarray]
    ) -> Dict[str, Any]:
        """
        Detects Non-ROI blocks from image path or numpy array.

        Returns:
            Dict containing:
                - 'header_block': (x1, y1, x2, y2)
                - 'payload_blocks': list of (x1, y1, x2, y2) tuples
                - 'all_blocks': list of all detected blocks
                - 'capacity_bytes': total available payload embedding capacity in bytes
                - 'header_capacity_bytes': capacity of the header block in bytes
                - 'image_shape': (height, width, channels)
                - 'roi_mask': binary mask (255 for ROI diagnostic zone, 0 for NROI)
        """
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Medical cover image not found: {image_input}")
            img = cv2.imread(image_input)
            if img is None:
                raise ValueError(f"Could not decode image at: {image_input}")
        elif isinstance(image_input, np.ndarray):
            img = image_input
        else:
            raise TypeError("image_input must be a file path string or numpy.ndarray")

        if len(img.shape) == 2:
            gray = img
            h, w = gray.shape
            channels = 1
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            h, w, channels = img.shape

        block_size = self.block_size
        detected_blocks: List[Tuple[int, int, int, int]] = []

        # 1. Block-based Non-ROI Detection Algorithm
        for y in range(0, h - block_size + 1, block_size):
            for x in range(0, w - block_size + 1, block_size):
                patch = gray[y : y + block_size, x : x + block_size]
                variance = float(np.var(patch))
                mean_val = float(np.mean(patch))

                # Identify low-variance / non-critical embedding zones
                is_uniform_dark = (mean_val < self.dark_mean_th and variance < self.dark_var_th)
                is_dense_white = (mean_val > self.dense_white_mean_th and variance < self.dense_white_var_th)
                is_low_detail = (variance < self.var_threshold and (mean_val < 70 or mean_val > 150))

                # Medical spatial zones: diaphragm base, mediastinum/spine, shoulders, axillary, corners
                in_bottom_zone = y > int(h * 0.72)
                in_spine_zone = (int(w * 0.44) < x < int(w * 0.54)) and (y > int(h * 0.35))
                in_corner_zone = (y < int(h * 0.15)) or (x < int(w * 0.15)) or (x > int(w * 0.85))
                in_retrocardiac = (int(w * 0.58) < x < int(w * 0.74)) and (int(h * 0.48) < y < int(h * 0.72))

                if (is_uniform_dark or is_dense_white or is_low_detail) and (
                    in_bottom_zone or in_spine_zone or in_corner_zone or in_retrocardiac
                ):
                    detected_blocks.append((x, y, x + block_size, y + block_size))

        # Always reserve top-left (0, 0, block_size, block_size) as the dedicated Header Block
        header_block = (0, 0, block_size, block_size)

        # Remove header block if present in detected_blocks to avoid overlap
        payload_blocks = [b for b in detected_blocks if b != header_block]

        # In case background detection is very sparse, ensure valid non-ROI peripheral blocks
        if not payload_blocks:
            # Fallback to peripheral corner blocks
            for y_fallback, x_fallback in [(0, w - block_size), (h - block_size, 0), (h - block_size, w - block_size)]:
                if (x_fallback, y_fallback, x_fallback + block_size, y_fallback + block_size) != header_block:
                    payload_blocks.append((x_fallback, y_fallback, x_fallback + block_size, y_fallback + block_size))

        # Capacity calculations (1 bit per pixel per channel for LSB)
        # Using 1-channel (grayscale/luminance) or multi-channel
        bits_per_pixel = channels  # e.g., 3 bits/pixel for BGR, 1 for grayscale
        bytes_per_block = (block_size * block_size * bits_per_pixel) // 8
        total_payload_capacity_bytes = len(payload_blocks) * bytes_per_block
        header_capacity_bytes = (block_size * block_size * bits_per_pixel) // 8

        # Create diagnostic ROI mask (255 = Diagnostic ROI, 0 = Non-ROI embedding area)
        roi_mask = np.ones((h, w), dtype=np.uint8) * 255
        # Clear header block
        roi_mask[0:block_size, 0:block_size] = 0
        # Clear payload blocks
        for (x1, y1, x2, y2) in payload_blocks:
            roi_mask[y1:y2, x1:x2] = 0

        return {
            "header_block": header_block,
            "payload_blocks": payload_blocks,
            "all_blocks": [header_block] + payload_blocks,
            "capacity_bytes": total_payload_capacity_bytes,
            "header_capacity_bytes": header_capacity_bytes,
            "image_shape": img.shape,
            "num_payload_blocks": len(payload_blocks),
            "bytes_per_block": bytes_per_block,
            "roi_mask": roi_mask,
        }

    def visualize_blocks(
        self,
        image_input: Union[str, np.ndarray],
        output_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Generates an annotated visual representation of detected blocks:
        - Red box on dedicated top-left header block
        - Green boxes on payload Non-ROI blocks
        - Diagnostic ROI preserved in original state
        """
        if isinstance(image_input, str):
            img = cv2.imread(image_input)
        else:
            img = image_input.copy()

        if len(img.shape) == 2:
            annotated = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            annotated = img.copy()

        det = self.detect_blocks(img)
        header_block = det["header_block"]
        payload_blocks = det["payload_blocks"]

        # Draw Green boxes for payload blocks
        for (x1, y1, x2, y2) in payload_blocks:
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 1)

        # Draw Red box for primary header block
        hx1, hy1, hx2, hy2 = header_block
        cv2.rectangle(annotated, (hx1, hy1), (hx2, hy2), (0, 0, 255), 2)

        # Add text badge
        stats_text = f"NROI Blocks: {len(payload_blocks)} | Capacity: {det['capacity_bytes']} B | Header: {det['header_capacity_bytes']} B"
        cv2.putText(
            annotated,
            stats_text,
            (10, annotated.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            cv2.imwrite(output_path, annotated)

        return annotated
