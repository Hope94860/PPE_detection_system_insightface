"""
SiteSecureVision — Vendor License Key Generator
================================================
Run this tool ONCE on your (the developer's) machine to:
  1. Generate an RSA-2048 keypair (first time only)
  2. Issue a signed .lic file for a paying customer

NEVER distribute private_key.pem — keep it secret.
The public_key.pem goes into license_manager.py.

Usage:
  python generate_license_tool.py --setup          # First time: generate keypair
  python generate_license_tool.py --issue           # Issue a new licence

Requirements:
  pip install cryptography
"""

import argparse
import base64
import hashlib
import json
from datetime import date, timedelta
from pathlib import Path

PRIVATE_KEY_FILE = Path("vendor_private_key.pem")
PUBLIC_KEY_FILE  = Path("vendor_public_key.pem")
_SALT = "SiteSecureVision@InvisibleFiction#2025"


def _mac_hash(mac: str) -> str:
    normalised = mac.strip().upper().replace("-", ":").replace(" ", "")
    raw = f"{_SALT}::{normalised}"
    digest = hashlib.sha256(raw.encode()).digest()
    return base64.b64encode(digest).decode()


def _payload_bytes(data: dict) -> bytes:
    signable = {
        "version":    data["version"],
        "customer":   data["customer"],
        "mac_hash":   data["mac_hash"],
        "issued_at":  data["issued_at"],
        "expires_at": data["expires_at"],
    }
    return json.dumps(signable, sort_keys=True, separators=(",", ":")).encode()


def setup_keypair():
    """Generate RSA-2048 keypair. Run once, keep private_key.pem secret."""
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.hazmat.primitives import serialization

    print("🔑 Generating RSA-2048 keypair...")
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )

    # Save private key (NEVER distribute this)
    PRIVATE_KEY_FILE.write_bytes(
        private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption()
        )
    )
    print(f"✅ Private key saved → {PRIVATE_KEY_FILE}  (KEEP THIS SECRET, NEVER SHARE)")

    # Save public key (embed this in license_manager.py)
    PUBLIC_KEY_FILE.write_bytes(
        private_key.public_key().private_bytes(  # type: ignore[attr-defined]
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        ) if hasattr(private_key.public_key(), 'private_bytes') else
        private_key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
    )
    # Fix: use public_bytes for public key
    PUBLIC_KEY_FILE.write_bytes(
        private_key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
    )
    print(f"✅ Public key saved  → {PUBLIC_KEY_FILE}")
    print()
    print("━" * 60)
    print("NEXT STEP: Copy the contents of vendor_public_key.pem into")
    print("license_manager.py → _PUBLIC_KEY_PEM variable, then rebuild")
    print("the installer.")
    print("━" * 60)


def issue_license():
    """Issue a signed .lic file for a paying customer."""
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding

    if not PRIVATE_KEY_FILE.exists():
        print("❌ Private key not found. Run --setup first.")
        return

    print("\n🔐 SiteSecureVision — Issue New Licence")
    print("━" * 40)

    customer = input("Customer / Organisation name: ").strip()
    if not customer:
        print("❌ Customer name required.")
        return

    mac = input("Customer's MAC address (e.g. 3C:22:FB:9D:04:A1): ").strip()
    if not mac or len(mac.replace(":", "").replace("-", "")) != 12:
        print("❌ Invalid MAC address format.")
        return

    expiry_input = input("Expiry date (YYYY-MM-DD) or press Enter for perpetual: ").strip()
    expires_at = expiry_input if expiry_input else None

    output_name = input(f"Output filename [sitesecurevision.lic]: ").strip() or "sitesecurevision.lic"

    # Build payload
    payload = {
        "version":    "2.0",
        "customer":   customer,
        "mac_hash":   _mac_hash(mac),
        "issued_at":  date.today().isoformat(),
        "expires_at": expires_at,
    }

    # Sign with private key
    private_key = serialization.load_pem_private_key(
        PRIVATE_KEY_FILE.read_bytes(),
        password=None,
    )
    signature = private_key.sign(  # type: ignore[union-attr]
        _payload_bytes(payload),
        padding.PKCS1v15(),
        hashes.SHA256()
    )
    payload["signature"] = base64.b64encode(signature).decode()

    # Write Base64-encoded .lic file
    raw_json = json.dumps(payload, indent=2)
    encoded = base64.b64encode(raw_json.encode()).decode()
    Path(output_name).write_text(encoded, encoding="utf-8")

    print()
    print("━" * 60)
    print(f"✅ Licence issued successfully!")
    print(f"   Customer : {customer}")
    print(f"   MAC      : {mac.upper()}")
    print(f"   Expires  : {expires_at or 'Perpetual (no expiry)'}")
    print(f"   File     : {output_name}")
    print()
    print(f"📧 Email '{output_name}' to the customer.")
    print("   They can import it using the SiteSecureVision Setup Wizard.")
    print("━" * 60)


def main():
    parser = argparse.ArgumentParser(description="SiteSecureVision Licence Tool")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--setup", action="store_true",
                       help="Generate RSA keypair (run once)")
    group.add_argument("--issue", action="store_true",
                       help="Issue a signed .lic file for a customer")
    args = parser.parse_args()

    if args.setup:
        setup_keypair()
    elif args.issue:
        issue_license()


if __name__ == "__main__":
    main()
