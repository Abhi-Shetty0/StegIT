"""
Clinical Cryptography Engine for Medical Steganography System (StegIT).

Provides authenticated encryption and decryption for sensitive Electronic Health
Record (EHR) JSON payloads using AES-256-GCM and ChaCha20-Poly1305.
"""

import os
import json
import struct
from typing import Dict, Any, Tuple, Optional
from cryptography.hazmat.primitives.ciphers.aead import AESGCM, ChaCha20Poly1305
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.exceptions import InvalidTag

MAGIC_BYTES = b"STEG"
CIPHER_AES_256_GCM = 0x01
CIPHER_CHACHA20_POLY1305 = 0x02

CIPHER_MAP = {
    "AES-256-GCM": CIPHER_AES_256_GCM,
    "AES-256": CIPHER_AES_256_GCM,
    "AES": CIPHER_AES_256_GCM,
    "ChaCha20-Poly1305": CIPHER_CHACHA20_POLY1305,
    "ChaCha20": CIPHER_CHACHA20_POLY1305,
    "CHACHA20": CIPHER_CHACHA20_POLY1305,
}

REVERSE_CIPHER_MAP = {
    CIPHER_AES_256_GCM: "AES-256-GCM",
    CIPHER_CHACHA20_POLY1305: "ChaCha20-Poly1305",
}

# Header struct: Magic(4s), CipherID(B), Salt(16s), Nonce(12s), CiphertextLength(I)
HEADER_STRUCT_FORMAT = ">4sB16s12sI"
HEADER_SIZE = struct.calcsize(HEADER_STRUCT_FORMAT)  # 4 + 1 + 16 + 12 + 4 = 37 bytes


class ClinicalCrypto:
    """
    Handles end-to-end cryptographic operations on patient EHR records.
    """

    def __init__(self, pbkdf2_iterations: int = 100_000):
        self.pbkdf2_iterations = pbkdf2_iterations

    def derive_key(self, passphrase: str, salt: bytes) -> bytes:
        """
        Derives a 256-bit (32-byte) key from the passphrase and salt using PBKDF2-HMAC-SHA256.
        """
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=self.pbkdf2_iterations,
        )
        return kdf.derive(passphrase.encode("utf-8"))

    def encrypt_bytes(
        self,
        plaintext_bytes: bytes,
        passphrase: str,
        cipher_type: str = "AES-256-GCM"
    ) -> Tuple[bytes, bytes, bytes]:
        """
        Encrypts arbitrary bytes using the specified AEAD cipher.
        
        Returns:
            Tuple of (header_bytes, ciphertext_bytes, complete_packet_bytes)
        """
        cipher_id = CIPHER_MAP.get(cipher_type, CIPHER_AES_256_GCM)
        salt = os.urandom(16)
        nonce = os.urandom(12)
        key = self.derive_key(passphrase, salt)

        if cipher_id == CIPHER_AES_256_GCM:
            aead = AESGCM(key)
        elif cipher_id == CIPHER_CHACHA20_POLY1305:
            aead = ChaCha20Poly1305(key)
        else:
            raise ValueError(f"Unsupported cipher type: {cipher_type}")

        ciphertext = aead.encrypt(nonce, plaintext_bytes, associated_data=None)
        payload_len = len(ciphertext)

        header_bytes = struct.pack(
            HEADER_STRUCT_FORMAT,
            MAGIC_BYTES,
            cipher_id,
            salt,
            nonce,
            payload_len
        )

        complete_packet = header_bytes + ciphertext
        return header_bytes, ciphertext, complete_packet

    def encrypt_json(
        self,
        payload_dict: Dict[str, Any],
        passphrase: str,
        cipher_type: str = "AES-256-GCM"
    ) -> Tuple[bytes, bytes, bytes]:
        """
        Serializes JSON dict to utf-8 and encrypts it.
        
        Returns:
            Tuple of (header_bytes, ciphertext_bytes, complete_packet_bytes)
        """
        json_bytes = json.dumps(payload_dict, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        return self.encrypt_bytes(json_bytes, passphrase, cipher_type)

    def decrypt_bytes(
        self,
        packet_bytes: bytes,
        passphrase: str,
        header_bytes: Optional[bytes] = None,
        ciphertext_bytes: Optional[bytes] = None
    ) -> bytes:
        """
        Decrypts an encrypted packet and authenticates the integrity of the data.
        Can accept either a full packet or separate header and ciphertext buffers.
        """
        if header_bytes is not None and ciphertext_bytes is not None:
            header_data = header_bytes
            ciphertext = ciphertext_bytes
        else:
            if len(packet_bytes) < HEADER_SIZE:
                raise ValueError(f"Packet size ({len(packet_bytes)} bytes) is smaller than required header size ({HEADER_SIZE} bytes).")
            header_data = packet_bytes[:HEADER_SIZE]
            ciphertext = packet_bytes[HEADER_SIZE:]

        magic, cipher_id, salt, nonce, payload_len = struct.unpack(HEADER_STRUCT_FORMAT, header_data)

        if magic != MAGIC_BYTES:
            raise ValueError(f"Invalid magic header bytes: {magic!r}. Expected {MAGIC_BYTES!r}.")

        if len(ciphertext) < payload_len:
            raise ValueError(f"Ciphertext truncated: received {len(ciphertext)} bytes, expected {payload_len} bytes.")

        # In case extra padding was provided
        ciphertext = ciphertext[:payload_len]

        key = self.derive_key(passphrase, salt)

        if cipher_id == CIPHER_AES_256_GCM:
            aead = AESGCM(key)
        elif cipher_id == CIPHER_CHACHA20_POLY1305:
            aead = ChaCha20Poly1305(key)
        else:
            raise ValueError(f"Unknown cipher ID: {cipher_id}")

        try:
            plaintext = aead.decrypt(nonce, ciphertext, associated_data=None)
            return plaintext
        except InvalidTag:
            raise ValueError("Decryption/Authentication failed: Incorrect password or corrupted steganographic payload.")

    def decrypt_json(
        self,
        packet_bytes: bytes,
        passphrase: str,
        header_bytes: Optional[bytes] = None,
        ciphertext_bytes: Optional[bytes] = None
    ) -> Dict[str, Any]:
        """
        Decrypts and deserializes ciphertext into structured JSON object.
        """
        plaintext = self.decrypt_bytes(packet_bytes, passphrase, header_bytes, ciphertext_bytes)
        return json.loads(plaintext.decode("utf-8"))

    @staticmethod
    def parse_header_metadata(header_bytes: bytes) -> Dict[str, Any]:
        """
        Parses header without requiring key/passphrase for diagnostic verification.
        """
        if len(header_bytes) < HEADER_SIZE:
            raise ValueError(f"Header length {len(header_bytes)} < {HEADER_SIZE}")
        magic, cipher_id, salt, nonce, payload_len = struct.unpack(HEADER_STRUCT_FORMAT, header_bytes[:HEADER_SIZE])
        return {
            "magic": magic.decode("latin-1", errors="replace"),
            "is_valid_magic": magic == MAGIC_BYTES,
            "cipher_id": cipher_id,
            "cipher_name": REVERSE_CIPHER_MAP.get(cipher_id, f"Unknown ({cipher_id})"),
            "salt_hex": salt.hex(),
            "nonce_hex": nonce.hex(),
            "payload_length_bytes": payload_len,
            "header_size_bytes": HEADER_SIZE,
            "total_packet_size_bytes": HEADER_SIZE + payload_len,
        }
