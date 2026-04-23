import PyPDF2
from typing import List, Dict, Any


class PDFProcessor:
    def __init__(self):
        pass

    def extract_pages(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract text from all pages of a PDF file"""
        pages = []

        try:
            with open(pdf_path, "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)

                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    pages.append(
                        {
                            "page_number": page_num + 1,
                            "text": text,
                            "char_count": len(text),
                        }
                    )

        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")

        return pages

    def get_page_chunks(self, pages: List[Dict], chunk_size: int = 20):
        """Split pages into chunks of specified size"""
        chunks = []
        for i in range(0, len(pages), chunk_size):
            chunk = pages[i:i + chunk_size]
            chunks.append(chunk)
        return chunks
