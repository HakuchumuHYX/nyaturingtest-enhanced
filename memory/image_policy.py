import re


MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_IMAGE_PIXELS = 4096 * 4096
MAX_CACHE_KEY_LEN = 128
MEM_CACHE_MAX_ITEMS = 512
MEM_CACHE_TTL_SECONDS = 6 * 60 * 60
SAFE_IMAGE_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}


def sanitize_image_cache_key(key: str | None) -> str | None:
    if not key:
        return None
    key = str(key).strip()
    if not key or len(key) > MAX_CACHE_KEY_LEN:
        return None
    if "/" in key or "\\" in key or ".." in key:
        return None
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", key):
        return None
    return key
