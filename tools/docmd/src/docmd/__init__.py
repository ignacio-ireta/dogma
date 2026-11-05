"""docmd — extract DOCX/PPTX/XLSX/PDF content into normalized Markdown."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("docmd")
except PackageNotFoundError:  # pragma: no cover - running from a raw checkout
    __version__ = "0.0.0+unknown"
