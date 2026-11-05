"""Tests for src/encryption. Run from the project root:
    python -m pytest tests/test_encryption.py
"""
import pytest

from src.encryption import (
    encrypt_bytes, decrypt_bytes, encrypt_json, decrypt_json,
)

PASSWORD = "test-password-123"
SAMPLE = {
    "patient": {"name": "John Doe", "age": 54},
    "entities": [
        {"text": "metformin", "label": "MEDICATION", "negated": False},
        {"text": "chest pain", "label": "SYMPTOM", "negated": True},
    ],
}


def test_bytes_round_trip():
    data = b"confidential clinical note"
    assert decrypt_bytes(encrypt_bytes(data, PASSWORD), PASSWORD) == data


def test_json_round_trip():
    assert decrypt_json(encrypt_json(SAMPLE, PASSWORD), PASSWORD) == SAMPLE


def test_unicode_survives():
    note = {"text": "Patient reports pain \u2013 \u00b5g dose, name: \u0c95\u0ca8\u0ccd\u0ca8\u0ca1"}
    assert decrypt_json(encrypt_json(note, PASSWORD), PASSWORD) == note


def test_wrong_password_rejected():
    blob = encrypt_json(SAMPLE, PASSWORD)
    with pytest.raises(ValueError):
        decrypt_json(blob, "wrong-password")


def test_tampered_ciphertext_rejected():
    blob = bytearray(encrypt_json(SAMPLE, PASSWORD))
    blob[-1] ^= 1
    with pytest.raises(ValueError):
        decrypt_json(bytes(blob), PASSWORD)


def test_tampered_header_rejected():
    blob = bytearray(encrypt_json(SAMPLE, PASSWORD))
    blob[10] ^= 1  # inside the salt
    with pytest.raises(ValueError):
        decrypt_json(bytes(blob), PASSWORD)


def test_not_a_stegit_blob_rejected():
    with pytest.raises(ValueError):
        decrypt_bytes(b"this is not an encrypted blob at all", PASSWORD)


def test_same_input_gives_different_output():
    # random salt + nonce each time
    assert encrypt_json(SAMPLE, PASSWORD) != encrypt_json(SAMPLE, PASSWORD)


def test_plaintext_not_visible_in_blob():
    blob = encrypt_json(SAMPLE, PASSWORD)
    assert b"John Doe" not in blob
    assert b"metformin" not in blob