"""Loader factory testleri."""
import pytest

from src.adapters.loaders import get_loader, UnsupportedFileType
from src.adapters.loaders.docx_loader import DocxLoader
from src.adapters.loaders.pdf_loader import PdfLoader
from src.adapters.loaders.txt_loader import TxtLoader
from src.adapters.loaders.doc_loader import DocLoader


def test_factory_returns_txt_loader_for_plain_text():
    assert isinstance(get_loader("text/plain"), TxtLoader)


def test_factory_returns_pdf_loader_for_pdf():
    assert isinstance(get_loader("application/pdf"), PdfLoader)


def test_factory_returns_docx_loader_for_docx():
    mime = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    assert isinstance(get_loader(mime), DocxLoader)


def test_factory_returns_doc_loader_for_doc():
    assert isinstance(get_loader("application/msword"), DocLoader)


def test_factory_by_extension_fallback():
    # MIME bilinmiyorsa filename ile eşleştir
    assert isinstance(get_loader(None, filename="report.pdf"), PdfLoader)
    assert isinstance(get_loader(None, filename="notes.txt"), TxtLoader)


def test_factory_unsupported_raises():
    with pytest.raises(UnsupportedFileType):
        get_loader("image/png", filename="screenshot.png")
    with pytest.raises(UnsupportedFileType):
        get_loader(None, filename="archive.zip")
