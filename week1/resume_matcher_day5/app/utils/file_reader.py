from pathlib import Path

from app.parsers.pdf_parser import read_pdf
from app.parsers.docx_parser import read_docx


def read_resume(file_path: Path):

    extension = file_path.suffix.lower()

    if extension == ".pdf":
        return read_pdf(file_path)

    if extension == ".docx":
        return read_docx(file_path)

    raise ValueError(f"Unsupported file type: {extension}")
