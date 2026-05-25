from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class BrowserTextBox:
    index: int
    text: str
    x: float
    y: float
    width: float
    height: float
    data_node_id: str | None = None

    def as_bbox(self):
        from .geometry import BBox

        return BBox(self.x, self.y, self.width, self.height)


class BrowserUnavailable(RuntimeError):
    pass


class BrowserSVGMeasurer:
    """Thin Playwright wrapper for rendered SVG text boxes.

    The class is deliberately optional. If Playwright/Chromium is not installed, callers can catch
    BrowserUnavailable and use XML fallback measurements.
    """

    def __init__(self, timeout_ms: int = 5000) -> None:
        self.timeout_ms = timeout_ms
        try:
            from playwright.sync_api import sync_playwright
        except Exception as e:  # pragma: no cover
            raise BrowserUnavailable(str(e)) from e
        self._sync_playwright = sync_playwright
        self._pw = None
        self._browser = None

    def __enter__(self):
        try:
            self._pw = self._sync_playwright().start()
            self._browser = self._pw.chromium.launch(headless=True)
            return self
        except Exception as e:  # pragma: no cover
            raise BrowserUnavailable(str(e)) from e

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._browser is not None:
            self._browser.close()
        if self._pw is not None:
            self._pw.stop()

    def measure_text(self, svg: str) -> list[BrowserTextBox]:
        if self._browser is None:
            raise BrowserUnavailable("browser not started")
        page = self._browser.new_page(viewport={"width": 1200, "height": 900})
        page.set_default_timeout(self.timeout_ms)
        data_url = "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("ascii")
        html = f"<html><body style='margin:0'><img id='svgimg' src='{data_url}'></body></html>"
        # Use inline SVG when possible so querySelectorAll('text') works.
        if "<svg" in svg:
            html = f"<html><body style='margin:0'>{svg}</body></html>"
        try:
            page.set_content(html, wait_until="load")
            rows: list[dict[str, Any]] = page.evaluate(
                """
                () => Array.from(document.querySelectorAll('text')).map((el, idx) => {
                  const b = el.getBBox();
                  return {
                    index: idx,
                    text: el.textContent || '',
                    x: b.x,
                    y: b.y,
                    width: b.width,
                    height: b.height,
                    data_node_id: el.getAttribute('data-node-id')
                  };
                })
                """
            )
            return [BrowserTextBox(**row) for row in rows]
        finally:
            page.close()
