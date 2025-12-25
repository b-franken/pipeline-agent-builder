import base64
import logging
import os
from pathlib import Path
from typing import Final

from cryptography.fernet import Fernet, InvalidToken

logger: Final = logging.getLogger(__name__)

_KEY_ENV_VAR: Final = "ENCRYPTION_KEY"
_KEY_FILE: Final = Path("./data/.encryption_key")

_fernet_instance: Fernet | None = None


def _get_or_create_key() -> bytes:
    env_key = os.environ.get(_KEY_ENV_VAR)
    if env_key:
        try:
            key = env_key.encode() if isinstance(env_key, str) else env_key
            Fernet(key)
            return key
        except (ValueError, InvalidToken):
            logger.warning(f"Invalid {_KEY_ENV_VAR} format, falling back to file")

    if _KEY_FILE.exists():
        try:
            key = _KEY_FILE.read_bytes().strip()
            Fernet(key)
            return key
        except (ValueError, InvalidToken):
            logger.warning("Invalid key file, generating new key")

    new_key: bytes = Fernet.generate_key()

    try:
        _KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
        _KEY_FILE.write_bytes(new_key)
        _KEY_FILE.chmod(0o600)
        logger.info(f"Generated new encryption key at {_KEY_FILE}")
    except OSError as e:
        logger.warning(f"Could not save encryption key to file: {e}")

    return new_key


def _get_fernet() -> Fernet:
    global _fernet_instance
    if _fernet_instance is None:
        key = _get_or_create_key()
        _fernet_instance = Fernet(key)
    return _fernet_instance


def encrypt(plaintext: str) -> str:
    if not plaintext:
        return ""

    fernet = _get_fernet()
    encrypted = fernet.encrypt(plaintext.encode("utf-8"))
    return base64.urlsafe_b64encode(encrypted).decode("ascii")


def decrypt(ciphertext: str) -> str:
    if not ciphertext:
        return ""

    try:
        fernet = _get_fernet()
        encrypted = base64.urlsafe_b64decode(ciphertext.encode("ascii"))
        decrypted: bytes = fernet.decrypt(encrypted)
        return decrypted.decode("utf-8")
    except InvalidToken as e:
        raise ValueError("Decryption failed: invalid key or corrupted data") from e
    except Exception as e:
        raise ValueError(f"Decryption failed: {e}") from e


def is_encrypted(value: str) -> bool:
    if not value or len(value) < 100:
        return False

    try:
        decoded = base64.urlsafe_b64decode(value.encode("ascii"))
        return len(decoded) >= 57 and decoded[0] == 0x80
    except Exception:
        return False
