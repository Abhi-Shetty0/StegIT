import json
import os
import base64

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


# ============================================================
# Configuration
# ============================================================

INPUT_FILE = "outputs/extracted/extracted_patient_data.json"
OUTPUT_FILE = "outputs/encrypted/encrypted_patient_data.json"
KEY_FILE = "keys/encryption_key.key"

NONCE_SIZE = 12
KEY_SIZE = 32


# ============================================================
# Key Management
# ============================================================

def generate_key():
    """
    Generate a new 256-bit AES encryption key.
    """

    return AESGCM.generate_key(bit_length=256)


def save_key(key):
    """
    Save the AES key to a local file.
    """

    os.makedirs(
        os.path.dirname(KEY_FILE),
        exist_ok=True
    )

    with open(KEY_FILE, "wb") as file:
        file.write(key)


def load_key():
    """
    Load the AES encryption key from the key file.
    """

    if not os.path.exists(KEY_FILE):
        raise FileNotFoundError(
            "Encryption key not found. "
            "Run the encryption process first."
        )

    with open(KEY_FILE, "rb") as file:
        return file.read()


# ============================================================
# AES-256-GCM Encryption
# ============================================================

def encrypt_record(record, key):
    """
    Encrypt one patient record using AES-256-GCM.

    The record is converted to JSON and then encrypted.
    """

    # Convert dictionary to JSON
    json_data = json.dumps(
        record,
        ensure_ascii=False,
        separators=(",", ":")
    )

    # Convert JSON to bytes
    plaintext = json_data.encode("utf-8")

    # Create AES-GCM object
    aesgcm = AESGCM(key)

    # Generate a unique random nonce
    nonce = os.urandom(NONCE_SIZE)

    # Encrypt the plaintext
    ciphertext = aesgcm.encrypt(
        nonce,
        plaintext,
        None
    )

    # Combine nonce and ciphertext
    encrypted_data = nonce + ciphertext

    # Convert binary data to Base64
    encoded_data = base64.b64encode(
        encrypted_data
    ).decode("utf-8")

    return encoded_data


# ============================================================
# AES-256-GCM Decryption
# ============================================================

def decrypt_record(encoded_data, key):
    """
    Decrypt one encrypted patient record.
    """

    # Decode Base64
    encrypted_data = base64.b64decode(
        encoded_data
    )

    # Separate nonce
    nonce = encrypted_data[:NONCE_SIZE]

    # Separate ciphertext
    ciphertext = encrypted_data[NONCE_SIZE:]

    # Create AES-GCM object
    aesgcm = AESGCM(key)

    # Decrypt
    plaintext = aesgcm.decrypt(
        nonce,
        ciphertext,
        None
    )

    # Convert bytes back to JSON
    json_data = plaintext.decode("utf-8")

    return json.loads(json_data)


# ============================================================
# Main Encryption Pipeline
# ============================================================

def encrypt_dataset():

    print("=" * 60)
    print("AES-256-GCM PATIENT DATA ENCRYPTION")
    print("=" * 60)

    # --------------------------------------------------------
    # Load extracted patient data
    # --------------------------------------------------------

    print("\nLoading extracted patient data...")

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        records = json.load(file)

    print(f"Records loaded: {len(records)}")

    # --------------------------------------------------------
    # Generate encryption key
    # --------------------------------------------------------

    print("\nGenerating AES-256 encryption key...")

    key = generate_key()

    save_key(key)

    print(f"Encryption key saved to: {KEY_FILE}")

    # --------------------------------------------------------
    # Encrypt records
    # --------------------------------------------------------

    encrypted_records = []

    print("\nEncrypting records...")

    for index, record in enumerate(records):

        encrypted_payload = encrypt_record(
            record,
            key
        )

        encrypted_record = {
            "record_id": record["record_id"],
            "indiana_uid": record["indiana_uid"],
            "image": record["image"],
            "image_path": record["image_path"],
            "encrypted_data": encrypted_payload
        }

        encrypted_records.append(
            encrypted_record
        )

        if (index + 1) % 500 == 0:
            print(
                f"Encrypted {index + 1}/{len(records)} records..."
            )

    # --------------------------------------------------------
    # Save encrypted dataset
    # --------------------------------------------------------

    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            encrypted_records,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("\n" + "=" * 60)
    print("ENCRYPTION COMPLETED")
    print("=" * 60)

    print(f"\nRecords encrypted: {len(encrypted_records)}")
    print(f"Output file: {OUTPUT_FILE}")


# ============================================================
# Test encryption and decryption
# ============================================================

def test_encryption():

    print("\n" + "=" * 60)
    print("TESTING ENCRYPTION / DECRYPTION")
    print("=" * 60)

    # Load original data
    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        records = json.load(file)

    original_record = records[0]

    # Load key
    key = load_key()

    # Encrypt
    encrypted_data = encrypt_record(
        original_record,
        key
    )

    print("\nOriginal record ID:")
    print(original_record["record_id"])

    print("\nEncrypted data:")
    print(encrypted_data[:100] + "...")

    # Decrypt
    decrypted_record = decrypt_record(
        encrypted_data,
        key
    )

    # Compare
    if decrypted_record == original_record:

        print("\nSUCCESS!")
        print(
            "Decrypted data matches the original data."
        )

    else:

        print("\nERROR!")
        print(
            "Decrypted data does not match "
            "the original data."
        )


# ============================================================
# Program Entry Point
# ============================================================

if __name__ == "__main__":

    encrypt_dataset()

    test_encryption()