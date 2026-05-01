import json
from pathlib import Path

from config import DATA_DIR, INDEX_CACHE_PATH
from schemas import RetrievedChunk


def infer_company_from_path(path: Path) -> str:
    root = path.parts[1].lower()
    mapping = {"claude": "Claude", "hackerrank": "HackerRank", "visa": "Visa"}
    return mapping[root]


def infer_product_area_from_path(path: Path) -> str:
    parts = list(path.parts[2:-1])
    return " / ".join(part.replace("-", " ") for part in parts) or "general"


def chunk_markdown_document(
    source_path: str,
    company: str,
    product_area_hint: str,
    markdown_text: str,
) -> list[RetrievedChunk]:
    title = ""
    section_heading = "overview"
    buffer: list[str] = []
    chunks: list[RetrievedChunk] = []

    def flush() -> None:
        if not buffer:
            return
        chunks.append(
            RetrievedChunk(
                company=company,
                product_area_hint=product_area_hint,
                source_path=source_path,
                title=title or Path(source_path).stem,
                section_heading=section_heading,
                chunk_text="\n".join(buffer).strip(),
            )
        )
        buffer.clear()

    for line in markdown_text.splitlines():
        if line.startswith("# "):
            flush()
            title = line[2:].strip()
            continue
        if line.startswith("## "):
            flush()
            section_heading = line[3:].strip()
            continue
        if line.strip():
            buffer.append(line.strip())

    flush()
    return chunks


def build_corpus_index() -> list[RetrievedChunk]:
    chunks: list[RetrievedChunk] = []
    for path in DATA_DIR.rglob("*.md"):
        markdown_text = path.read_text(encoding="utf-8")
        relative_path = path.relative_to(DATA_DIR.parent).as_posix()
        relative_obj = Path(relative_path)
        company = infer_company_from_path(relative_obj)
        product_area_hint = infer_product_area_from_path(relative_obj)
        chunks.extend(
            chunk_markdown_document(
                source_path=relative_path,
                company=company,
                product_area_hint=product_area_hint,
                markdown_text=markdown_text,
            )
        )
    return chunks


def save_index(chunks: list[RetrievedChunk]) -> None:
    INDEX_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = [chunk.model_dump() for chunk in chunks]
    INDEX_CACHE_PATH.write_text(json.dumps(payload), encoding="utf-8")


def load_or_build_index(force_rebuild: bool = False) -> list[RetrievedChunk]:
    if INDEX_CACHE_PATH.exists() and not force_rebuild:
        data = json.loads(INDEX_CACHE_PATH.read_text(encoding="utf-8"))
        return [RetrievedChunk(**item) for item in data]

    chunks = build_corpus_index()
    save_index(chunks)
    return chunks
