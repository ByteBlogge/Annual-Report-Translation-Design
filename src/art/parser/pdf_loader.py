"""Input loading: PDF -> page images, or image files -> pages.

Zero-dependency image dimension probing
---------------------------------------
Rasterising a PDF needs PyMuPDF, but *reading a PNG's width and height* does
not need Pillow -- those bytes are at a fixed offset in the header. Since the
SDO needs page dimensions for geometry, implementing a ~60-line header reader
keeps the whole image-input path on the standard library. Pillow is then only
needed if you want to re-encode or crop, which the pipeline does not.

Rasterisation DPI matters more than it looks: at 96 DPI, hairline table rules
and 6pt footnote digits disappear, and the table structure model sees merged
columns that are not there. 180 DPI is the practical floor for financial
statements, and the default here.
"""

from __future__ import annotations

import struct
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .analyzer import PageSource

__all__ = [
    "image_size",
    "load_pages",
    "rasterise_pdf",
    "SUPPORTED_IMAGE_SUFFIXES",
    "PDF_INSTALL_HINT",
]

SUPPORTED_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tif", ".tiff"})

PDF_INSTALL_HINT = (
    "PDF input needs PyMuPDF. Install it with:\n"
    "    pip install -e \".[pdf]\"\n"
    "Alternatively, pre-render the pages to PNG (for example `pdftoppm -r 180 -png in.pdf page`) "
    "and pass the images directly -- everything downstream is identical."
)


# ---------------------------------------------------------------------------
# image header probing
# ---------------------------------------------------------------------------


def image_size(data: bytes) -> tuple[int, int] | None:
    """Return ``(width, height)`` for common image formats, or ``None``."""
    if len(data) < 24:
        return None
    if data[:8] == b"\x89PNG\r\n\x1a\n" and data[12:16] == b"IHDR":
        return struct.unpack(">II", data[16:24])
    if data[:2] == b"\xff\xd8":
        return _jpeg_size(data)
    if data[:6] in (b"GIF87a", b"GIF89a"):
        w, h = struct.unpack("<HH", data[6:10])
        return w, h
    if data[:2] == b"BM" and len(data) >= 26:
        w, h = struct.unpack("<ii", data[18:26])
        return abs(w), abs(h)
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return _webp_size(data)
    return None


def _jpeg_size(data: bytes) -> tuple[int, int] | None:
    """Walk JPEG segments to the Start-Of-Frame marker."""
    i = 2
    n = len(data)
    sof_markers = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
    while i + 9 < n:
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        if i + 4 > n:
            return None
        length = struct.unpack(">H", data[i + 2 : i + 4])[0]
        if marker in sof_markers:
            height, width = struct.unpack(">HH", data[i + 5 : i + 9])
            return width, height
        if length < 2:
            return None
        i += 2 + length
    return None


def _webp_size(data: bytes) -> tuple[int, int] | None:
    chunk = data[12:16]
    if chunk == b"VP8X" and len(data) >= 30:
        w = int.from_bytes(data[24:27], "little") + 1
        h = int.from_bytes(data[27:30], "little") + 1
        return w, h
    if chunk == b"VP8L" and len(data) >= 25:
        bits = int.from_bytes(data[21:25], "little")
        return (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1
    if chunk == b"VP8 " and len(data) >= 30:
        w = int.from_bytes(data[26:28], "little") & 0x3FFF
        h = int.from_bytes(data[28:30], "little") & 0x3FFF
        return w, h
    return None


# ---------------------------------------------------------------------------
# page loading
# ---------------------------------------------------------------------------


@dataclass
class PageLoadOptions:
    dpi: int = 180
    #: Include the PDF's embedded text layer when present. Cheap, and it gives
    #: the VLM exact wording to correct OCR errors against.
    include_text_layer: bool = True
    max_pages: int | None = None
    first_page: int = 1  # 1-based, PDF convention


def load_pages(source: str | Path, options: PageLoadOptions | None = None) -> list[PageSource]:
    """Load a PDF or image path (or directory of images) into page sources."""
    opts = options or PageLoadOptions()
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"input not found: {path}")
    if path.is_dir():
        return _load_directory(path, opts)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return rasterise_pdf(path, opts)
    if suffix in SUPPORTED_IMAGE_SUFFIXES:
        return [_load_single_image(path, opts)]
    raise ValueError(f"unsupported input type {suffix!r}; expected .pdf or one of {sorted(SUPPORTED_IMAGE_SUFFIXES)}")


def _load_directory(path: Path, opts: PageLoadOptions) -> list[PageSource]:
    files = sorted(
        (p for p in path.iterdir() if p.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES),
        key=lambda p: p.name,
    )
    if opts.max_pages is not None:
        files = files[: opts.max_pages]
    return [_load_single_image(p, opts, index=i) for i, p in enumerate(files)]


def _load_single_image(path: Path, opts: PageLoadOptions, index: int = 0) -> PageSource:
    data = path.read_bytes()
    size = image_size(data)
    width, height = (float(size[0]), float(size[1])) if size else (0.0, 0.0)
    mime = "image/jpeg" if path.suffix.lower() in (".jpg", ".jpeg") else f"image/{path.suffix.lower().lstrip('.')}"
    return PageSource(
        page_index=index,
        width=width,
        height=height,
        image=data,
        image_path=str(path),
        metadata={"mime": mime, "filename": path.name},
    )


def rasterise_pdf(path: Path, opts: PageLoadOptions | None = None) -> list[PageSource]:
    """Render PDF pages to PNG bytes with PyMuPDF."""
    opts = opts or PageLoadOptions()
    try:
        import fitz  # type: ignore[import-not-found]  # PyMuPDF
    except ImportError as exc:
        raise ImportError(PDF_INSTALL_HINT) from exc

    pages: list[PageSource] = []
    with fitz.open(path) as document:
        total = document.page_count
        start = max(1, opts.first_page) - 1
        stop = total if opts.max_pages is None else min(total, start + opts.max_pages)
        zoom = opts.dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        for index in range(start, stop):
            page = document.load_page(index)
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            text_layer = page.get_text("text") if opts.include_text_layer else ""
            pages.append(
                PageSource(
                    page_index=index + 1,
                    width=pixmap.width / zoom if zoom else float(pixmap.width),
                    height=pixmap.height / zoom if zoom else float(pixmap.height),
                    image=pixmap.tobytes("png"),
                    text_layer=text_layer,
                    metadata={
                        "mime": "image/png",
                        "dpi": opts.dpi,
                        "pdf_page": index + 1,
                        "rotated": page.rotation,
                    },
                )
            )
    return pages


def iter_page_batches(pages: list[PageSource], size: int) -> Iterator[list[PageSource]]:
    """Yield fixed-size batches, for analyzers that can process several pages per call."""
    size = max(1, size)
    for i in range(0, len(pages), size):
        yield pages[i : i + size]


def summarise_pages(pages: list[PageSource]) -> dict[str, Any]:
    return {
        "pages": len(pages),
        "with_image": sum(1 for p in pages if p.has_image),
        "with_text_layer": sum(1 for p in pages if p.text_layer.strip()),
        "first_size": [pages[0].width, pages[0].height] if pages else None,
    }
