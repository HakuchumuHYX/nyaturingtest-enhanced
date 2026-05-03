import json

from nonebot import logger


def log_event(event: str, **fields):
    payload = {"event": event}
    payload.update({key: value for key, value in fields.items() if value is not None})
    logger.info(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
