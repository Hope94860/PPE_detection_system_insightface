"""
License / Configuration Manager for SiteSecureVision
=====================================================
SECURE IMPLEMENTATION — RSA-Signed License Files
-------------------------------------------------
How it works:
  1. The VENDOR (you) holds a private RSA key (never distributed).
  2. When a customer pays, you run generate_license_tool.py on your machine,
     providing the customer's MAC address → this produces a signed .lic file.
  3. The .lic file is emailed to the customer.
  4. The customer places the .lic file in the app folder (or Setup Wizard guides them).
  5. On every launch, the app verifies the RSA signature using the PUBLIC key
     embedded in this file. Without your private key, a valid .lic cannot be forged.

The Setup Wizard no longer generates the .lic itself — it only asks the
user to PROVIDE a .lic file that was issued by the vendor.
"""

from __future__ import annotations

import json
import hashlib
import base64
import uuid
import platform
from pathlib import Path
from datetime import datetime, date
from typing import Optional, Tuple

# ─── Paths ───────────────────────────────────────────────────────────────────
import os
import platform

def _get_app_data_dir() -> Path:
    if platform.system() == "Windows":
        base_dir = Path(os.environ.get("APPDATA", Path.home()))
    else:
        base_dir = Path.home() / ".config"
    app_dir = base_dir / "SiteSecureVision"
    app_dir.mkdir(parents=True, exist_ok=True)
    return app_dir

BASE_DIR    = Path(__file__).parent
APP_DATA_DIR = _get_app_data_dir()
CONFIG_FILE = APP_DATA_DIR / "sitesecurevision.lic"

# ─── Embedded Public Key (RSA 2048-bit, PEM format) ──────────────────────────
# This is the PUBLIC key only. The private key NEVER leaves the vendor's machine.
# Run generate_license_tool.py once to generate your keypair, then paste the
# public key PEM here and rebuild the installer.
#
_PUBLIC_KEY_PEM = """-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA9G10O5HFLrvuQBm9Mh0S
NlKF73KaPf8G2lN8GioIsrL3nRLL/COkK6TmUvNaHx3tYY+yo0ClYcguLeofbN9n
TzzdHXrb40auizBH3XWqT9C0mwtFC2AwnUPP4VVLD+5Qsv+em/PCe4AtRiabyE9S
LongjM+MtYXfNfMkrUp+dUiQcHWUIzIy208lK5yyA6cLMmZfjQODepzAmlrJwwjF
wLfHDsTnw0oTIzd18/Ks9iERgsG/tZWDfAgF2LfMUlDgutecHuXXghDLvh+VWaXW
5cunOes6Uhsli29Jyi60vCqYAZfe1K7LUJo7qzbeMXXAcsYyH+7byfgfPOQfOCEC
PQIDAQAB
-----END PUBLIC KEY-----"""

# ─── MAC helpers ─────────────────────────────────────────────────────────────

def _get_current_mac() -> Optional[str]:
    """Return the primary non-loopback MAC address of this machine."""
    try:
        mac_int = uuid.getnode()
        if (mac_int >> 40) & 1:   # randomly generated flag
            return None
        parts = []
        for i in range(5, -1, -1):
            parts.append(f"{(mac_int >> (8 * i)) & 0xFF:02X}")
        return ":".join(parts)
    except Exception:
        return None


def _all_macs() -> list[str]:
    """Return every MAC-like string we can find on this machine."""
    macs = set()
    mac = _get_current_mac()
    if mac:
        macs.add(mac.upper().replace("-", ":"))
    try:
        import subprocess, re
        if platform.system() == "Windows":
            # Use ipconfig /all — always available, unlike getmac which may not be on PATH
            out = subprocess.check_output(
                ["ipconfig", "/all"], shell=False, stderr=subprocess.DEVNULL
            ).decode(errors="replace")
            for line in out.splitlines():
                line = line.strip()
                if "Physical Address" in line:
                    # Format: "Physical Address. . . . . . . . : AA-BB-CC-DD-EE-FF"
                    m = re.search(r"([0-9A-Fa-f]{2}[-:][0-9A-Fa-f]{2}[-:][0-9A-Fa-f]{2}[-:][0-9A-Fa-f]{2}[-:][0-9A-Fa-f]{2}[-:][0-9A-Fa-f]{2})", line)
                    if m:
                        raw = m.group(1).replace("-", ":").upper()
                        macs.add(raw)
        elif platform.system() in ("Linux", "Darwin"):
            out = subprocess.check_output(
                ["ip", "link", "show"], shell=False, stderr=subprocess.DEVNULL
            ).decode(errors="replace")
            for line in out.splitlines():
                line = line.strip()
                if line.startswith("link/ether"):
                    m = line.split()[1].upper()
                    macs.add(m)
    except Exception:
        pass
    return list(macs)


def _normalise_mac(mac: str) -> str:
    return mac.strip().upper().replace("-", ":").replace(" ", "")


# ─── License file structure ───────────────────────────────────────────────────
#
#  The .lic file is a Base64-encoded JSON blob:
#  {
#    "version":    "2.0",
#    "customer":   "Acme Corp",
#    "mac_hash":   "<SHA-256(salt::MAC)>",      # which machine is licensed
#    "issued_at":  "2025-03-20",
#    "expires_at": "2026-03-20" | null,          # null = perpetual
#    "signature":  "<RSA-SHA256 signature of the above fields, base64>"
#  }
#
#  The VENDOR creates this file with generate_license_tool.py.
#  The APP verifies the signature; it cannot create a valid file.

_SALT = "SiteSecureVision@InvisibleFiction#2025"


def _mac_hash(mac: str) -> str:
    normalised = _normalise_mac(mac)
    raw = f"{_SALT}::{normalised}"
    digest = hashlib.sha256(raw.encode()).digest()
    return base64.b64encode(digest).decode()


def _payload_bytes(data: dict) -> bytes:
    """Deterministic serialisation of the signable fields."""
    signable = {
        "version":    data["version"],
        "customer":   data["customer"],
        "mac_hash":   data["mac_hash"],
        "issued_at":  data["issued_at"],
        "expires_at": data["expires_at"],
    }
    return json.dumps(signable, sort_keys=True, separators=(",", ":")).encode()


def _verify_signature(data: dict) -> bool:
    """
    Verify the RSA-SHA256 signature embedded in the .lic payload.
    Returns True only if the signature is valid and was made with our private key.
    """
    try:
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import padding
        from cryptography.hazmat.backends import default_backend

        public_key = serialization.load_pem_public_key(
            _PUBLIC_KEY_PEM.strip().encode(),
            backend=default_backend()
        )
        sig = base64.b64decode(data["signature"].encode())
        public_key.verify(
            sig,
            _payload_bytes(data),
            padding.PKCS1v15(),
            hashes.SHA256()
        )
        return True
    except Exception as e:
        print(f"[License] Signature verification failed: {e}")
        return False


# ─── Public API ───────────────────────────────────────────────────────────────

def is_configured() -> bool:
    """Return True if a licence file is present (not yet validated)."""
    return CONFIG_FILE.exists()


def load_license() -> Optional[dict]:
    """Decode and return the .lic payload dict, or None if missing/corrupt."""
    if not CONFIG_FILE.exists():
        return None
    try:
        raw_b64 = CONFIG_FILE.read_text(encoding="utf-8").strip()
        raw_json = base64.b64decode(raw_b64.encode()).decode()
        return json.loads(raw_json)
    except Exception as e:
        print(f"[License] Failed to read licence file: {e}")
        return None


def validate_license() -> Tuple[bool, str]:
    """
    Full licence validation:
      1. File present and decodable
      2. RSA signature valid (cannot be forged without vendor's private key)
      3. MAC address of this machine matches the licensed MAC hash
      4. Licence not expired

    Returns (is_valid: bool, message: str)
    """
    data = load_license()
    if data is None:
        return False, (
            "No valid licence file found.\n\n"
            "Please contact Invisible Fiction to obtain a licence\n"
            "for this machine, then place sitesecurevision.lic\n"
            "in the application folder."
        )

    # 1 — Signature check (most important — cannot be faked)
    if not _verify_signature(data):
        return False, (
            "Licence file is invalid or has been tampered with.\n\n"
            "Please contact Invisible Fiction for a replacement licence."
        )

    # 2 — Expiry check
    expires = data.get("expires_at")
    if expires:
        try:
            exp_date = date.fromisoformat(expires)
            if date.today() > exp_date:
                return False, (
                    f"Your licence expired on {expires}.\n\n"
                    "Please contact Invisible Fiction to renew your licence."
                )
        except ValueError:
            pass

    # 3 — MAC address check
    stored_hash = data.get("mac_hash", "")
    current_macs = _all_macs()
    if not current_macs:
        return False, "Could not read this machine's MAC address."

    for mac in current_macs:
        if _mac_hash(mac) == stored_hash:
            customer = data.get("customer", "Unknown")
            return True, f"Licence valid — Licensed to: {customer} (MAC: {mac})"

    return False, (
        "This licence is not valid for this machine.\n\n"
        "Your licence is issued to a different machine.\n"
        "Please contact Invisible Fiction to transfer your licence."
    )


# Keep old name as alias so launcher.py doesn't need to change
def validate_mac() -> Tuple[bool, str]:
    return validate_license()


def get_cameras_from_config() -> list[dict]:
    """Return camera list stored in config.json (cameras are no longer in .lic)."""
    config_path = APP_DATA_DIR / "config.json"
    if not config_path.exists():
        return []
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f).get("cameras", [])
    except Exception:
        return []


def save_cameras(cameras: list[dict]) -> bool:
    """Save camera configuration to config.json (separate from the licence)."""
    try:
        config_path = APP_DATA_DIR / "config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump({"cameras": cameras}, f, indent=2)
        return True
    except Exception as e:
        print(f"[License] Failed to save cameras: {e}")
        return False
