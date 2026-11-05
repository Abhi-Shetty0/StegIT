"""
src/encryption.py  --  Encryption layer for StegIT
AES-256-GCM (authenticated encryption) + scrypt password-based key derivation.

Blob layout (all bytes, easy to embed in an image later):
    MAGIC(4) | VERSION(1) | SALT(16) | NONCE(12) | CIPHERTEXT+TAG(n+16)
"""
import json
import os
import zlib
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
from cryptography.exceptions import InvalidTag

MAGIC = b"STGT"
VERSION = b"\x01"
SALT_LEN, NONCE_LEN = 16, 12
HEADER_LEN = len(MAGIC) + len(VERSION) + SALT_LEN + NONCE_LEN


def _derive_key(password: str, salt: bytes) -> bytes:
    kdf = Scrypt(salt=salt, length=32, n=2**15, r=8, p=1)
    return kdf.derive(password.encode("utf-8"))


def encrypt_bytes(data: bytes, password: str) -> bytes:
    salt, nonce = os.urandom(SALT_LEN), os.urandom(NONCE_LEN)
    key = _derive_key(password, salt)
    header = MAGIC + VERSION + salt + nonce
    # header is authenticated too (AAD) so tampering with it is detected
    ct = AESGCM(key).encrypt(nonce, data, header)
    return header + ct


def decrypt_bytes(blob: bytes, password: str) -> bytes:
    if blob[:4] != MAGIC:
        raise ValueError("Not a StegIT encrypted blob")
    salt = blob[5:5 + SALT_LEN]
    nonce = blob[5 + SALT_LEN:HEADER_LEN]
    header = blob[:HEADER_LEN]
    key = _derive_key(password, salt)
    try:
        return AESGCM(key).decrypt(nonce, blob[HEADER_LEN:], header)
    except InvalidTag:
        raise ValueError("Wrong password or corrupted/tampered data")


# ---- Convenience wrappers for the pipeline's structured JSON output ----
# Compress BEFORE encrypting (ciphertext looks random and cannot be compressed).
# Smaller payload = less image capacity needed for the steganography stage.
def encrypt_json(result: dict, password: str) -> bytes:
    raw = json.dumps(result, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return encrypt_bytes(zlib.compress(raw, 9), password)


def decrypt_json(blob: bytes, password: str) -> dict:
    return json.loads(zlib.decompress(decrypt_bytes(blob, password)).decode("utf-8"))


def encrypt_file(in_path: str, out_path: str, password: str) -> None:
    with open(in_path, "rb") as f:
        blob = encrypt_bytes(f.read(), password)
    with open(out_path, "wb") as f:
        f.write(blob)


def decrypt_file(in_path: str, out_path: str, password: str) -> None:
    with open(in_path, "rb") as f:
        data = decrypt_bytes(f.read(), password)
    with open(out_path, "wb") as f:
        f.write(data)


if __name__ == "__main__":
    sample = {"patient": {"age": 54}, "entities": [{"text": "metformin", "label": "MEDICATION", "negated": False}]}
    blob = encrypt_json(sample, "demo-password")
    assert decrypt_json(blob, "demo-password") == sample
    try:
        decrypt_json(blob, "wrong")
    except ValueError as e:
        print("Wrong password rejected:", e)
    t = bytearray(blob); t[-1] ^= 1
    try:
        decrypt_json(bytes(t), "demo-password")
    except ValueError as e:
        print("Tampering rejected:", e)
    print("Round-trip OK, blob size:", len(blob), "bytes")