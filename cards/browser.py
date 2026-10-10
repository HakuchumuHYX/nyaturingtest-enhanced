"""卡片截图：把填好的 html 交给 Chromium 截长图。

看卡片不常用，内存又紧，所以每次现开浏览器、截完就关，不常驻。
"""

import os

from playwright.async_api import async_playwright

from ..config import get_app_settings


async def screenshot(html: str, width: int) -> bytes:
    # 和 HakuBot 共用同一份 Chromium（playwright 版本一致），不用再下载
    browsers_path = get_app_settings().playwright_browsers_path
    if browsers_path:
        os.environ["PLAYWRIGHT_BROWSERS_PATH"] = browsers_path
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(args=["--no-sandbox"])
        try:
            page = await browser.new_page(
                viewport={"width": width, "height": 200}, device_scale_factor=2
            )
            await page.set_content(html)
            await page.evaluate("document.fonts.ready")
            return await page.screenshot(full_page=True)
        finally:
            await browser.close()
