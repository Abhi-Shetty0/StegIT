import json
import os
import struct
import zlib
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


# =========================================================
# CONFIGURATION
# =========================================================

INPUT_FILE = Path(
    "outputs/extracted/extracted_patient_data.json"
)

OUTPUT_FILE = Path(
    "outputs/encrypted/binary_payload.bin"
)

KEY_FILE = Path(
    "keys/encryption_key.key"
)

# AES-GCM uses a 12-byte nonce
NONCE_SIZE = 12

# 32-bit unsigned integer = 4 bytes
HEADER_SIZE = 4


# =========================================================
# JSON → UTF-8 BYTES
# =========================================================

def serialize_json(data):
    """
    Convert Python JSON data into UTF-8 encoded bytes.
    """

    json_string = json.dumps(
        data,
        ensure_ascii=False,
        separators=(",", ":")
    )

    return json_string.encode("utf-8")


# =========================================================
# UTF-8 BYTES → JSON
# =========================================================

def deserialize_json(data):
    """
    Convert UTF-8 encoded bytes back into Python JSON data.
    """

    json_string = data.decode("utf-8")

    return json.loads(json_string)


# =========================================================
# ZLIB COMPRESSION
# =========================================================

def compress_data(data):
    """
    Compress data using zlib.
    """

    return zlib.compress(data, level=9)


# =========================================================
# ZLIB DECOMPRESSION
# =========================================================

def decompress_data(data):
    """
    Decompress zlib-compressed data.
    """

    return zlib.decompress(data)


# =========================================================
# AES-256-GCM ENCRYPTION
# =========================================================

def encrypt_data(data, key):
    """
    Encrypt data using AES-256-GCM.

    Output format:

        [12-byte nonce][encrypted data + authentication tag]
    """

    if len(key) != 32:
        raise ValueError(
            "AES key must be exactly 32 bytes (256 bits)."
        )

    aesgcm = AESGCM(key)

    # Generate a cryptographically secure random 12-byte nonce
    nonce = os.urandom(NONCE_SIZE)

    encrypted_data = aesgcm.encrypt(
        nonce,
        data,
        None
    )

    return nonce + encrypted_data


# =========================================================
# AES-256-GCM DECRYPTION
# =========================================================

def decrypt_data(data, key):
    """
    Decrypt AES-256-GCM data.

    Expected format:

        [12-byte nonce][encrypted data + authentication tag]
    """

    if len(key) != 32:
        raise ValueError(
            "AES key must be exactly 32 bytes (256 bits)."
        )

    if len(data) <= NONCE_SIZE:
        raise ValueError(
            "Encrypted data is too small."
        )

    # Extract nonce
    nonce = data[:NONCE_SIZE]

    # Extract ciphertext + authentication tag
    encrypted_data = data[NONCE_SIZE:]

    aesgcm = AESGCM(key)

    decrypted_data = aesgcm.decrypt(
        nonce,
        encrypted_data,
        None
    )

    return decrypted_data


# =========================================================
# 32-BIT PAYLOAD LENGTH HEADER
# =========================================================

def package_payload(payload):
    """
    Add a 4-byte header containing the exact payload length.

    Format:

        [4-byte payload length][payload]

    The >I format means:
        >  = big-endian
        I  = unsigned 32-bit integer
    """

    payload_length = len(payload)

    # Maximum value supported by unsigned 32-bit integer
    if payload_length > 0xFFFFFFFF:
        raise ValueError(
            "Payload is too large for a 32-bit length header."
        )

    header = struct.pack(
        ">I",
        payload_length
    )

    return header + payload


# =========================================================
# UNPACK PAYLOAD
# =========================================================

def unpack_payload(data):
    """
    Read the 4-byte length header and extract
    exactly the specified number of payload bytes.
    """

    # Make sure the data contains at least the header
    if len(data) < HEADER_SIZE:
        raise ValueError(
            "Payload is too small to contain a 4-byte header."
        )

    # Read the payload length
    payload_length = struct.unpack(
        ">I",
        data[:HEADER_SIZE]
    )[0]

    # Extract payload
    payload = data[HEADER_SIZE:]

    # Verify exact payload length
    if len(payload) != payload_length:
        raise ValueError(
            f"Payload length mismatch. "
            f"Expected {payload_length} bytes, "
            f"but found {len(payload)} bytes."
        )

    return payload


# =========================================================
# COMPLETE PIPELINE TEST
# =========================================================

def test_pipeline():

    print("\n" + "=" * 65)
    print("SERIALIZATION + COMPRESSION + ENCRYPTION PIPELINE")
    print("=" * 65)

    # -----------------------------------------------------
    # 1. LOAD STRUCTURED JSON
    # -----------------------------------------------------

    print("\n[1] Loading structured clinical JSON...")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    print(f"Records loaded: {len(data)}")

    if len(data) == 0:
        raise ValueError(
            "The extracted JSON file contains no records."
        )

    # Use first record for testing
    original_record = data[0]

    print(
        f"Testing record ID: "
        f"{original_record.get('record_id')}"
    )

    # -----------------------------------------------------
    # 2. JSON → UTF-8
    # -----------------------------------------------------

    serialized = serialize_json(
        original_record
    )

    print("\n[2] JSON Serialization")
    print(
        f"Original JSON size: "
        f"{len(serialized)} bytes"
    )

    # -----------------------------------------------------
    # 3. ZLIB COMPRESSION
    # -----------------------------------------------------

    compressed = compress_data(
        serialized
    )

    print("\n[3] zlib Compression")
    print(
        f"Compressed size: "
        f"{len(compressed)} bytes"
    )

    compression_percentage = (
        len(compressed)
        / len(serialized)
    ) * 100

    reduction_percentage = (
        1
        - (
            len(compressed)
            / len(serialized)
        )
    ) * 100

    print(
        f"Compressed to: "
        f"{compression_percentage:.2f}% "
        f"of original"
    )

    print(
        f"Size reduction: "
        f"{reduction_percentage:.2f}%"
    )

    # -----------------------------------------------------
    # 4. LOAD AES-256 KEY
    # -----------------------------------------------------

    print("\n[4] Loading AES-256 encryption key...")

    if not KEY_FILE.exists():
        raise FileNotFoundError(
            f"Encryption key not found: {KEY_FILE}\n"
            "Run the AES encryption stage first."
        )

    with open(
        KEY_FILE,
        "rb"
    ) as f:

        key = f.read()

    print(
        f"Key size: "
        f"{len(key) * 8} bits"
    )

    if len(key) != 32:
        raise ValueError(
            "The encryption key is not 256 bits."
        )

    # -----------------------------------------------------
    # 5. AES-256-GCM ENCRYPTION
    # -----------------------------------------------------

    encrypted = encrypt_data(
        compressed,
        key
    )

    print("\n[5] AES-256-GCM Encryption")

    print(
        f"Encrypted payload size: "
        f"{len(encrypted)} bytes"
    )

    print(
        f"Nonce size: "
        f"{NONCE_SIZE} bytes"
    )

    # -----------------------------------------------------
    # 6. ADD 32-BIT LENGTH HEADER
    # -----------------------------------------------------

    packaged = package_payload(
        encrypted
    )

    print("\n[6] 32-bit Payload Length Header")

    print(
        f"Header size: "
        f"{HEADER_SIZE} bytes"
    )

    print(
        f"Payload size: "
        f"{len(encrypted)} bytes"
    )

    print(
        f"Total binary size: "
        f"{len(packaged)} bytes"
    )

    # -----------------------------------------------------
    # 7. SAVE BINARY PAYLOAD
    # -----------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "wb"
    ) as f:

        f.write(packaged)

    print("\n[7] Binary Payload Saved")

    print(
        f"File: {OUTPUT_FILE}"
    )

    print(
        f"File size: "
        f"{OUTPUT_FILE.stat().st_size} bytes"
    )

    # =====================================================
    # RECOVERY / VERIFICATION TEST
    # =====================================================

    print("\n" + "-" * 65)
    print("RECOVERY AND VERIFICATION TEST")
    print("-" * 65)

    # -----------------------------------------------------
    # 8. READ BINARY FILE
    # -----------------------------------------------------

    print("\n[8] Reading binary payload...")

    with open(
        OUTPUT_FILE,
        "rb"
    ) as f:

        packaged_data = f.read()

    print(
        f"Binary data read: "
        f"{len(packaged_data)} bytes"
    )

    # -----------------------------------------------------
    # 9. EXTRACT PAYLOAD USING HEADER
    # -----------------------------------------------------

    extracted_encrypted = unpack_payload(
        packaged_data
    )

    print("\n[9] Header / Payload Extraction")

    print(
        f"Extracted encrypted payload: "
        f"{len(extracted_encrypted)} bytes"
    )

    # -----------------------------------------------------
    # 10. AES DECRYPTION
    # -----------------------------------------------------

    decrypted_compressed = decrypt_data(
        extracted_encrypted,
        key
    )

    print("\n[10] AES-256-GCM Decryption")

    print(
        f"Decrypted compressed data: "
        f"{len(decrypted_compressed)} bytes"
    )

    # -----------------------------------------------------
    # 11. ZLIB DECOMPRESSION
    # -----------------------------------------------------

    decompressed = decompress_data(
        decrypted_compressed
    )

    print("\n[11] zlib Decompression")

    print(
        f"Recovered JSON bytes: "
        f"{len(decompressed)} bytes"
    )

    # -----------------------------------------------------
    # 12. UTF-8 → JSON
    # -----------------------------------------------------

    recovered_record = deserialize_json(
        decompressed
    )

    print("\n[12] JSON Deserialization")

    print(
        "Structured JSON successfully recovered."
    )

    # -----------------------------------------------------
    # 13. EXACT COMPARISON
    # -----------------------------------------------------

    print("\n[13] Final Verification")

    if recovered_record == original_record:

        print("\n" + "=" * 65)
        print("SUCCESS!")
        print("=" * 65)

        print(
            "Original and recovered JSON are EXACTLY identical."
        )

        print("\nVerification Results:")
        print("  Serialization          : PASSED")
        print("  zlib Compression       : PASSED")
        print("  AES-256-GCM Encryption : PASSED")
        print("  32-bit Length Header   : PASSED")
        print("  Binary Packaging       : PASSED")
        print("  Payload Extraction     : PASSED")
        print("  AES-256-GCM Decryption : PASSED")
        print("  zlib Decompression     : PASSED")
        print("  JSON Recovery          : PASSED")
        print("  Exact Data Comparison  : PASSED")

        print("=" * 65)

    else:

        print("\n" + "=" * 65)
        print("FAILED!")
        print("=" * 65)

        print(
            "The recovered JSON does NOT match "
            "the original JSON."
        )

        print("=" * 65)


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    try:

        test_pipeline()

    except Exception as e:

        print("\n" + "=" * 65)
        print("ERROR")
        print("=" * 65)
        print(type(e).__name__ + ":", e)
        print("=" * 65)