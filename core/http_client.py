import ssl

import httpx
from nonebot import logger


_HTTP_CLIENT: httpx.AsyncClient | None = None


def get_http_client() -> httpx.AsyncClient:
    """Return the process-wide pooled HTTP client."""

    global _HTTP_CLIENT
    if _HTTP_CLIENT is None:
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
        ssl_context.set_ciphers("ALL:@SECLEVEL=1")
        _HTTP_CLIENT = httpx.AsyncClient(
            verify=ssl_context,
            timeout=30.0,
            limits=httpx.Limits(
                max_keepalive_connections=50,
                max_connections=100,
            ),
        )
    return _HTTP_CLIENT


async def close_http_client() -> None:
    global _HTTP_CLIENT
    if _HTTP_CLIENT is not None:
        await _HTTP_CLIENT.aclose()
        _HTTP_CLIENT = None
        logger.info("全局 HTTP 客户端已关闭")
