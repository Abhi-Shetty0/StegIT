"""
Non-ROI Least Significant Bit (LSB) Steganography Module for StegIT.

Performs payload capacity verification, cryptographic packet embedding into
detected Non-ROI background blocks, and blind extraction/decryption of EHR data
while preserving clinical diagnostic zones.
"""

import os
from typing import Dict, Any, Tuple, Optional, Union
import cv2
import numpy as np

from src.crypto import ClinicalCrypto, HEADER_SIZE, MAGIC_BYTES
from src.roi_detector import NonROIDetector


def _bytes_to_bits(data: bytes) -> np.ndarray:
    """Converts a byte array into a 1D numpy array of bits (uint8)."""
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8))


def _bits_to_bytes(bits: np.ndarray) -> bytes:
    """Converts a 1D numpy array of bits into bytes."""
    # Ensure length is multiple of 8
    pad_len = (8 - (len(bits) % 8)) % 8
    if pad_len > 0:
        bits = np.pad(bits, (0, pad_len), mode="constant")
    return np.packbits(bits).tobytes()


class LSBSteganography:
    """
    Handles LSB embedding and extraction for medical cover images.
    """

    def __init__(
        self,
        block_size: int = 48,
        crypto: Optional[ClinicalCrypto] = None,
        roi_detector: Optional[NonROIDetector] = None,
    ):
        self.block_size = block_size
        self.crypto = crypto or ClinicalCrypto()
        self.roi_detector = roi_detector or NonROIDetector(block_size=block_size)

    def embed_payload(
        self,
        cover_image_input: Union[str, np.ndarray],
        payload_data: Union[Dict[str, Any], str, bytes],
        passphrase: str,
        cipher_type: str = "AES-256-GCM",
        output_stego_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Encrypts and embeds EHR payload into Non-ROI blocks of the cover image.

        Args:
            cover_image_input: Path to cover image or numpy array.
            payload_data: Dict (JSON EHR), str, or raw bytes to hide.
            passphrase: Secret key / password for AES-256 / ChaCha20 encryption.
            cipher_type: 'AES-256-GCM' or 'ChaCha20-Poly1305'.
            output_stego_path: Optional path to save stego PNG.

        Returns:
            Dict containing stego image, metadata, block stats, and capacity info.
        """
        # 1. Load cover image
        if isinstance(cover_image_input, str):
            if not os.path.exists(cover_image_input):
                raise FileNotFoundError(f"Cover image not found: {cover_image_input}")
            cover_img = cv2.imread(cover_image_input)
            if cover_img is None:
                raise ValueError(f"Failed to read image at: {cover_image_input}")
        else:
            cover_img = cover_image_input.copy()

        stego_img = cover_img.copy()
        is_grayscale = len(stego_img.shape) == 2
        channels = 1 if is_grayscale else stego_img.shape[2]

        # 2. Detect Non-ROI blocks
        det_info = self.roi_detector.detect_blocks(cover_img)
        header_block = det_info["header_block"]
        payload_blocks = det_info["payload_blocks"]
        total_capacity_bytes = det_info["capacity_bytes"]
        header_capacity_bytes = det_info["header_capacity_bytes"]

        # 3. Encrypt payload
        if isinstance(payload_data, dict):
            header_bytes, ciphertext_bytes, complete_packet = self.crypto.encrypt_json(
                payload_data, passphrase, cipher_type
            )
        elif isinstance(payload_data, str):
            header_bytes, ciphertext_bytes, complete_packet = self.crypto.encrypt_bytes(
                payload_data.encode("utf-8"), passphrase, cipher_type
            )
        elif isinstance(payload_data, bytes):
            header_bytes, ciphertext_bytes, complete_packet = self.crypto.encrypt_bytes(
                payload_data, passphrase, cipher_type
            )
        else:
            raise TypeError("payload_data must be dict, str, or bytes")

        # 4. Capacity Checks
        if len(header_bytes) > header_capacity_bytes:
            raise ValueError(
                f"Header size ({len(header_bytes)} bytes) exceeds header block capacity ({header_capacity_bytes} bytes)."
            )

        if len(ciphertext_bytes) > total_capacity_bytes:
            raise ValueError(
                f"Ciphertext size ({len(ciphertext_bytes)} bytes) exceeds total Non-ROI payload capacity ({total_capacity_bytes} bytes). "
                f"Need {len(ciphertext_bytes)} bytes, but only {total_capacity_bytes} bytes available in {len(payload_blocks)} blocks."
            )

        # 5. Embed Header into dedicated top-left Header Block
        header_bits = _bytes_to_bits(header_bytes)
        hx1, hy1, hx2, hy2 = header_block
        header_bit_idx = 0
        total_header_bits = len(header_bits)

        for y in range(hy1, hy2):
            for x in range(hx1, hx2):
                if is_grayscale:
                    if header_bit_idx < total_header_bits:
                        bit = header_bits[header_bit_idx]
                        stego_img[y, x] = (stego_img[y, x] & ~1) | bit
                        header_bit_idx += 1
                else:
                    for c in range(channels):
                        if header_bit_idx < total_header_bits:
                            bit = header_bits[header_bit_idx]
                            stego_img[y, x, c] = (stego_img[y, x, c] & ~1) | bit
                            header_bit_idx += 1

        # 6. Embed Ciphertext sequentially across Green Non-ROI Blocks
        payload_bits = _bytes_to_bits(ciphertext_bytes)
        payload_bit_idx = 0
        total_payload_bits = len(payload_bits)
        blocks_used = 0

        for (bx1, by1, bx2, by2) in payload_blocks:
            if payload_bit_idx >= total_payload_bits:
                break
            blocks_used += 1
            for y in range(by1, by2):
                for x in range(bx1, bx2):
                    if is_grayscale:
                        if payload_bit_idx < total_payload_bits:
                            bit = payload_bits[payload_bit_idx]
                            stego_img[y, x] = (stego_img[y, x] & ~1) | bit
                            payload_bit_idx += 1
                    else:
                        for c in range(channels):
                            if payload_bit_idx < total_payload_bits:
                                bit = payload_bits[payload_bit_idx]
                                stego_img[y, x, c] = (stego_img[y, x, c] & ~1) | bit
                                payload_bit_idx += 1

        # 7. Save Stego Image if requested (always as lossless PNG)
        if output_stego_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_stego_path)), exist_ok=True)
            # Ensure .png extension for lossless storage
            if not output_stego_path.lower().endswith(".png"):
                output_stego_path = os.path.splitext(output_stego_path)[0] + ".png"
            cv2.imwrite(output_stego_path, stego_img)

        return {
            "stego_image": stego_img,
            "cover_image": cover_img,
            "output_path": output_stego_path,
            "cipher_type": cipher_type,
            "header_bytes_len": len(header_bytes),
            "ciphertext_bytes_len": len(ciphertext_bytes),
            "total_capacity_bytes": total_capacity_bytes,
            "capacity_utilization_pct": (len(ciphertext_bytes) / total_capacity_bytes) * 100.0 if total_capacity_bytes > 0 else 0.0,
            "num_payload_blocks": len(payload_blocks),
            "blocks_used": blocks_used,
            "header_block": header_block,
            "payload_blocks": payload_blocks,
            "roi_mask": det_info["roi_mask"],
        }

    def extract_payload(
        self,
        stego_image_input: Union[str, np.ndarray],
        passphrase: str,
    ) -> Dict[str, Any]:
        """
        Performs blind extraction of metadata header and encrypted EHR ciphertext
        from the stego image, authenticating and decrypting with the passphrase.

        Returns:
            Dict containing:
                - 'decrypted_payload': Decrypted JSON dict or string
                - 'header_metadata': Header info (cipher type, length, salt, nonce)
                - 'is_verified': True if cryptographic tag authenticated
                - 'extracted_bytes_len': Total ciphertext bytes extracted
        """
        # 1. Load stego image
        if isinstance(stego_image_input, str):
            if not os.path.exists(stego_image_input):
                raise FileNotFoundError(f"Stego image not found: {stego_image_input}")
            stego_img = cv2.imread(stego_image_input)
            if stego_img is None:
                raise ValueError(f"Failed to read image at: {stego_image_input}")
        else:
            stego_img = stego_image_input.copy()

        is_grayscale = len(stego_img.shape) == 2
        channels = 1 if is_grayscale else stego_img.shape[2]

        # 2. Extract Header Block (top-left)
        hx1, hy1, hx2, hy2 = (0, 0, self.block_size, self.block_size)
        total_header_bits = HEADER_SIZE * 8
        header_bits = []

        for y in range(hy1, hy2):
            for x in range(hx1, hx2):
                if len(header_bits) >= total_header_bits:
                    break
                if is_grayscale:
                    header_bits.append(int(stego_img[y, x] & 1))
                else:
                    for c in range(channels):
                        if len(header_bits) >= total_header_bits:
                            break
                        header_bits.append(int(stego_img[y, x, c] & 1))

        header_bytes = _bits_to_bytes(np.array(header_bits, dtype=np.uint8))[:HEADER_SIZE]
        header_meta = self.crypto.parse_header_metadata(header_bytes)

        if not header_meta["is_valid_magic"]:
            raise ValueError(
                f"Stego header validation failed. Magic bytes '{header_meta['magic']}' do not match expected '{MAGIC_BYTES.decode()}'. "
                "The image may not contain StegIT steganographic data."
            )

        payload_len = header_meta["payload_length_bytes"]
        total_payload_bits = payload_len * 8

        # 3. Detect Non-ROI payload blocks
        det_info = self.roi_detector.detect_blocks(stego_img)
        payload_blocks = det_info["payload_blocks"]

        # 4. Extract Ciphertext bits from Non-ROI blocks
        payload_bits = []
        for (bx1, by1, bx2, by2) in payload_blocks:
            if len(payload_bits) >= total_payload_bits:
                break
            for y in range(by1, by2):
                for x in range(bx1, bx2):
                    if len(payload_bits) >= total_payload_bits:
                        break
                    if is_grayscale:
                        payload_bits.append(int(stego_img[y, x] & 1))
                    else:
                        for c in range(channels):
                            if len(payload_bits) >= total_payload_bits:
                                break
                            payload_bits.append(int(stego_img[y, x, c] & 1))

        if len(payload_bits) < total_payload_bits:
            raise ValueError(
                f"Extraction error: Expected {total_payload_bits} bits ({payload_len} bytes), but only extracted {len(payload_bits)} bits from Non-ROI blocks."
            )

        ciphertext_bytes = _bits_to_bytes(np.array(payload_bits, dtype=np.uint8))[:payload_len]

        # 5. Decrypt and verify payload
        decrypted_json = self.crypto.decrypt_json(
            packet_bytes=b"",
            passphrase=passphrase,
            header_bytes=header_bytes,
            ciphertext_bytes=ciphertext_bytes,
        )

        return {
            "decrypted_payload": decrypted_json,
            "header_metadata": header_meta,
            "is_verified": True,
            "extracted_bytes_len": len(ciphertext_bytes),
        }
