import os
import json
import base64
import zstandard as zstd
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

PAYLOADS_DIR = "extracted_payloads"
OUTPUT_FILE = "encrypted_dataset.json"

class MedicalDataEncryptor:
    def __init__(self, key: bytes = None):
        # 256-bit (32 bytes) symmetric key
        self.key = key if key else ChaCha20Poly1305.generate_key()
        self.cipher = ChaCha20Poly1305(self.key)
        self.compressor = zstd.ZstdCompressor(level=10)
        self.decompressor = zstd.ZstdDecompressor()

    def encrypt_payload(self, clinical_data: dict) -> str:
        # Step 1: JSON to Bytes
        raw_bytes = json.dumps(clinical_data).encode("utf-8")

        # Step 2: Zstd Compression
        compressed_bytes = self.compressor.compress(raw_bytes)

        # Step 3: 12-byte random Nonce
        nonce = os.urandom(12)

        # Step 4: ChaCha20-Poly1305 AEAD Encryption
        ciphertext_with_tag = self.cipher.encrypt(nonce, compressed_bytes, associated_data=None)

        # Step 5: Pack Nonce + Ciphertext + Tag
        final_binary_payload = nonce + ciphertext_with_tag

        # Return Base64 encoded string
        return base64.b64encode(final_binary_payload).decode("utf-8")

    def decrypt_payload(self, base64_payload: str) -> dict:
        encrypted_bytes = base64.b64decode(base64_payload.encode("utf-8"))
        
        nonce = encrypted_bytes[:12]
        ciphertext_with_tag = encrypted_bytes[12:]
        
        decrypted_compressed = self.cipher.decrypt(nonce, ciphertext_with_tag, associated_data=None)
        decompressed_bytes = self.decompressor.decompress(decrypted_compressed)
        
        return json.loads(decompressed_bytes.decode("utf-8"))

def main():
    print("=" * 70)
    print("  BATCH ENCRYPTION PIPELINE: ENCRYPTING ALL PATIENTS (Zstd + ChaCha20)")
    print("=" * 70)

    if not os.path.exists(PAYLOADS_DIR):
        print(f"❌ Error: '{PAYLOADS_DIR}' directory not found! Make sure you are in the StegIT root folder.")
        return

    encryptor = MedicalDataEncryptor()
    
    # Save the secret key so you can decrypt anytime
    key_hex = encryptor.key.hex()
    with open("encryption_key.txt", "w") as f:
        f.write(key_hex)
    print(f"🔑 Symmetric Key (Saved to 'encryption_key.txt'): {key_hex}\n")

    # Find and sort all patient payload files
    payload_files = [f for f in os.listdir(PAYLOADS_DIR) if f.startswith("patient_") and f.endswith(".json")]
    
    # Sort numerically by patient ID
    def extract_id(filename):
        try:
            return int(filename.split("_")[1])
        except (IndexError, ValueError):
            return 999999
            
    payload_files.sort(key=extract_id)
    total_files = len(payload_files)
    print(f"📂 Found {total_files} patient payloads in '{PAYLOADS_DIR}'. Starting encryption...\n")

    encrypted_dataset = []

    for index, filename in enumerate(payload_files, 1):
        filepath = os.path.join(PAYLOADS_DIR, filename)
        
        with open(filepath, "r", encoding="utf-8") as f:
            patient_json = json.load(f)

        patient_id = extract_id(filename)
        image_name = patient_json.get("mapped_cover_image", f"cover_{patient_id}.png")
        
        # Encrypt the entire clinical payload
        encrypted_str = encryptor.encrypt_payload(patient_json)

        encrypted_dataset.append({
            "record_id": index,
            "patient_id": patient_id,
            "image": os.path.basename(image_name),
            "image_path": image_name,
            "encrypted_data": encrypted_str
        })

        if index % 500 == 0 or index == total_files:
            print(f"  [✓] Encrypted {index}/{total_files} patients...")

    # Write all encrypted records to encrypted_dataset.json
    print(f"\n💾 Saving all records to '{OUTPUT_FILE}'...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(encrypted_dataset, f, indent=4)

    print(f"🎉 SUCCESS! All {total_files} patient records encrypted and saved to '{OUTPUT_FILE}'.")

    # Verify random record decryption
    print("\n" + "=" * 70)
    print("  VERIFYING DECRYPTION (Testing Record 1)")
    print("=" * 70)
    test_decrypted = encryptor.decrypt_payload(encrypted_dataset[0]["encrypted_data"])
    print(f"Demographics for Patient {encrypted_dataset[0]['patient_id']}:", test_decrypted.get("patient_demographics"))
    print("✅ Decryption and Decompression verified successfully!")

if __name__ == "__main__":
    main()