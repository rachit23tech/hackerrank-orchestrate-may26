from pathlib import Path

from corpus import chunk_markdown_document, infer_company_from_path


def test_infer_company_from_path():
    path = Path("data/claude/claude-mobile-apps/general/file.md")
    assert infer_company_from_path(path) == "Claude"


def test_chunk_markdown_document_preserves_headings():
    text = "# Title\n\n## Billing\n\nLine one.\nLine two.\n\n## Security\n\nLine three."
    chunks = chunk_markdown_document(
        source_path="data/claude/example.md",
        company="Claude",
        product_area_hint="billing",
        markdown_text=text,
    )

    assert len(chunks) >= 2
    assert chunks[0].title == "Title"
    assert chunks[0].section_heading == "Billing"
    assert "Line one" in chunks[0].chunk_text
