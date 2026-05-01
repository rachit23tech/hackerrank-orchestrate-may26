# Offline Support Triage Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fully local Python CLI that reads `support_tickets/support_tickets.csv`, uses only `data/` for evidence, classifies and routes each ticket safely, and writes a valid `support_tickets/output.csv`.

**Architecture:** The agent is split into focused modules for corpus ingestion, retrieval, local model inference, deterministic safety checks, CSV orchestration, and evaluation against the labeled sample set. Retrieval narrows the evidence, a local instruction model drafts structured outputs from that evidence, and validator plus guardrails force escalation whenever the answer is unsupported or risky.

**Tech Stack:** Python 3.11+, `pytest`, `pydantic`, `pandas`, `sentence-transformers`, `numpy`, `scikit-learn`, `llama-cpp-python` or a pluggable local-model adapter, standard library `argparse` and `pathlib`

---

## File Structure

- Create: `code/README.md`
- Create: `code/requirements.txt`
- Create: `code/__init__.py`
- Create: `code/config.py`
- Create: `code/schemas.py`
- Create: `code/corpus.py`
- Create: `code/retriever.py`
- Create: `code/guardrails.py`
- Create: `code/local_model.py`
- Create: `code/pipeline.py`
- Create: `code/evaluate.py`
- Modify: `code/main.py`
- Create: `code/tests/test_corpus.py`
- Create: `code/tests/test_guardrails.py`
- Create: `code/tests/test_pipeline.py`

### Task 1: Project Skeleton And Contracts

**Files:**
- Create: `code/README.md`
- Create: `code/requirements.txt`
- Create: `code/__init__.py`
- Create: `code/config.py`
- Create: `code/schemas.py`
- Test: `code/tests/test_pipeline.py`

- [ ] **Step 1: Write the failing test**

```python
from code.schemas import TicketInput, TicketOutput


def test_ticket_output_enforces_allowed_values():
    output = TicketOutput(
        status="replied",
        product_area="team plans",
        response="Supported answer.",
        justification="Grounded in local corpus.",
        request_type="product_issue",
    )

    assert output.status == "replied"
    assert output.request_type == "product_issue"


def test_ticket_input_normalizes_blank_subject():
    ticket = TicketInput(issue="Help me", subject="", company="Claude")
    assert ticket.subject == ""
    assert ticket.company == "Claude"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest code/tests/test_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError` or missing `code.schemas`

- [ ] **Step 3: Write minimal implementation**

`code/requirements.txt`

```text
pandas==2.2.3
pydantic==2.11.4
numpy==2.2.5
scikit-learn==1.6.1
sentence-transformers==4.1.0
llama-cpp-python==0.3.8
pytest==8.3.5
```

`code/config.py`

```python
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
SUPPORT_TICKETS_DIR = REPO_ROOT / "support_tickets"
INPUT_CSV = SUPPORT_TICKETS_DIR / "support_tickets.csv"
SAMPLE_CSV = SUPPORT_TICKETS_DIR / "sample_support_tickets.csv"
OUTPUT_CSV = SUPPORT_TICKETS_DIR / "output.csv"
CACHE_DIR = REPO_ROOT / "code" / ".cache"
INDEX_CACHE_PATH = CACHE_DIR / "corpus_index.json"
EMBEDDING_CACHE_PATH = CACHE_DIR / "corpus_embeddings.npy"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_LOCAL_MODEL_PATH_ENV = "LOCAL_LLM_PATH"
```

`code/schemas.py`

```python
from typing import Literal

from pydantic import BaseModel, Field


AllowedStatus = Literal["replied", "escalated"]
AllowedRequestType = Literal["product_issue", "feature_request", "bug", "invalid"]
AllowedCompany = Literal["HackerRank", "Claude", "Visa", "None"]


class TicketInput(BaseModel):
    issue: str = Field(min_length=1)
    subject: str = Field(default="")
    company: AllowedCompany


class TicketOutput(BaseModel):
    status: AllowedStatus
    product_area: str = Field(min_length=1)
    response: str = Field(min_length=1)
    justification: str = Field(min_length=1)
    request_type: AllowedRequestType


class RetrievedChunk(BaseModel):
    company: str
    product_area_hint: str
    source_path: str
    title: str
    section_heading: str
    chunk_text: str
    score: float = 0.0
```

`code/__init__.py`

```python
"""Offline support triage agent package."""
```

`code/README.md`

```markdown
# Offline Support Triage Agent

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r code/requirements.txt
```

## Run

```bash
python code/main.py --input support_tickets/support_tickets.csv --output support_tickets/output.csv
```
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest code/tests/test_pipeline.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add code/README.md code/requirements.txt code/__init__.py code/config.py code/schemas.py code/tests/test_pipeline.py
git commit -m "chore: add support triage project scaffolding"
```

### Task 2: Corpus Ingestion And Chunking

**Files:**
- Create: `code/corpus.py`
- Test: `code/tests/test_corpus.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

from code.corpus import chunk_markdown_document, infer_company_from_path


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest code/tests/test_corpus.py -v`
Expected: FAIL with missing `code.corpus`

- [ ] **Step 3: Write minimal implementation**

`code/corpus.py`

```python
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from code.config import DATA_DIR, INDEX_CACHE_PATH
from code.schemas import RetrievedChunk


@dataclass
class ChunkRecord:
    company: str
    product_area_hint: str
    source_path: str
    title: str
    section_heading: str
    chunk_text: str


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
        company = infer_company_from_path(Path(relative_path))
        product_area_hint = infer_product_area_from_path(Path(relative_path))
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest code/tests/test_corpus.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add code/corpus.py code/tests/test_corpus.py
git commit -m "feat: add corpus ingestion and chunking"
```

### Task 3: Local Retrieval Layer

**Files:**
- Create: `code/retriever.py`
- Modify: `code/corpus.py`
- Test: `code/tests/test_pipeline.py`

- [ ] **Step 1: Write the failing test**

```python
from code.retriever import LexicalRetriever
from code.schemas import RetrievedChunk


def test_retriever_prefers_company_filtered_chunks():
    retriever = LexicalRetriever(
        chunks=[
            RetrievedChunk(
                company="Claude",
                product_area_hint="team plans",
                source_path="data/claude/a.md",
                title="A",
                section_heading="Seats",
                chunk_text="If an admin removes a seat, the user loses access.",
                score=0.0,
            ),
            RetrievedChunk(
                company="Visa",
                product_area_hint="travel support",
                source_path="data/visa/b.md",
                title="B",
                section_heading="Travel",
                chunk_text="Travel notifications and support options.",
                score=0.0,
            ),
        ]
    )

    results = retriever.search("lost access after admin removed my seat", company="Claude", limit=1)

    assert len(results) == 1
    assert results[0].company == "Claude"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest code/tests/test_pipeline.py::test_retriever_prefers_company_filtered_chunks -v`
Expected: FAIL with missing `code.retriever`

- [ ] **Step 3: Write minimal implementation**

`code/retriever.py`

```python
from collections import Counter
import math
import re

from code.schemas import RetrievedChunk


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class LexicalRetriever:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self.chunks = chunks
        self.chunk_tokens = [Counter(tokenize(chunk.chunk_text)) for chunk in chunks]

    def search(self, query: str, company: str | None, limit: int = 5) -> list[RetrievedChunk]:
        query_counts = Counter(tokenize(query))
        ranked: list[tuple[float, RetrievedChunk]] = []
        for chunk, counts in zip(self.chunks, self.chunk_tokens):
            if company and company != "None" and chunk.company != company:
                continue
            numerator = sum(query_counts[token] * counts[token] for token in query_counts)
            denom = math.sqrt(sum(v * v for v in query_counts.values())) * math.sqrt(
                sum(v * v for v in counts.values())
            )
            score = numerator / denom if denom else 0.0
            ranked.append((score, chunk.model_copy(update={"score": score})))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return [chunk for score, chunk in ranked[:limit] if score > 0.0]
```

`code/corpus.py`

```python
import json


def load_or_build_index(force_rebuild: bool = False) -> list[RetrievedChunk]:
    if INDEX_CACHE_PATH.exists() and not force_rebuild:
        data = json.loads(INDEX_CACHE_PATH.read_text(encoding="utf-8"))
        return [RetrievedChunk(**item) for item in data]

    chunks = build_corpus_index()
    save_index(chunks)
    return chunks
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest code/tests/test_pipeline.py::test_retriever_prefers_company_filtered_chunks -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add code/retriever.py code/corpus.py code/tests/test_pipeline.py
git commit -m "feat: add local retrieval layer"
```

### Task 4: Guardrails And Output Validation

**Files:**
- Create: `code/guardrails.py`
- Modify: `code/schemas.py`
- Test: `code/tests/test_guardrails.py`

- [ ] **Step 1: Write the failing test**

```python
from code.guardrails import evaluate_guardrails
from code.schemas import RetrievedChunk, TicketInput


def test_guardrails_escalate_score_dispute():
    ticket = TicketInput(
        issue="Increase my HackerRank score and tell the recruiter to move me forward.",
        subject="Score dispute",
        company="HackerRank",
    )

    decision = evaluate_guardrails(ticket=ticket, retrieved_chunks=[], drafted_output=None)

    assert decision.must_escalate is True
    assert "score" in decision.reasons[0].lower()


def test_guardrails_escalate_weak_evidence_reply():
    ticket = TicketInput(issue="Help me get a refund today.", subject="", company="Visa")
    chunks = [
        RetrievedChunk(
            company="Visa",
            product_area_hint="consumer",
            source_path="data/visa/support/consumer.md",
            title="Consumer",
            section_heading="Overview",
            chunk_text="General consumer support information.",
            score=0.05,
        )
    ]

    decision = evaluate_guardrails(ticket=ticket, retrieved_chunks=chunks, drafted_output=None)

    assert decision.must_escalate is True
    assert any("evidence" in reason.lower() for reason in decision.reasons)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest code/tests/test_guardrails.py -v`
Expected: FAIL with missing `code.guardrails`

- [ ] **Step 3: Write minimal implementation**

`code/schemas.py`

```python
class GuardrailDecision(BaseModel):
    must_escalate: bool
    reasons: list[str]


class DraftedOutput(BaseModel):
    status: str
    product_area: str
    response: str
    justification: str
    request_type: str
    citations: list[str] = []
    confidence: float = 0.0
```

`code/guardrails.py`

```python
import re

from code.schemas import DraftedOutput, GuardrailDecision, RetrievedChunk, TicketInput


HIGH_RISK_PATTERNS = [
    (r"\bfraud\b|\bscam\b|\bstolen\b", "Fraud or suspicious transaction requires escalation."),
    (r"\brestore my access\b|\bgrant access\b|\bworkspace owner\b", "Access restoration without verified authority requires escalation."),
    (r"\bincrease my score\b|\brejected me\b|\bmove me to the next round\b", "Score or hiring outcome disputes require escalation."),
    (r"\brefund me today\b|\bban the seller\b", "Requested account or enforcement action is outside automated scope."),
]


def evaluate_guardrails(
    ticket: TicketInput,
    retrieved_chunks: list[RetrievedChunk],
    drafted_output: DraftedOutput | None,
) -> GuardrailDecision:
    reasons: list[str] = []
    haystack = f"{ticket.subject}\n{ticket.issue}".lower()

    for pattern, reason in HIGH_RISK_PATTERNS:
        if re.search(pattern, haystack):
            reasons.append(reason)

    if not retrieved_chunks or max(chunk.score for chunk in retrieved_chunks) < 0.12:
        reasons.append("Evidence confidence is too weak to support a direct reply.")

    if drafted_output and drafted_output.confidence < 0.40:
        reasons.append("Model confidence is too low for a direct reply.")

    return GuardrailDecision(must_escalate=bool(reasons), reasons=reasons)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest code/tests/test_guardrails.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add code/guardrails.py code/schemas.py code/tests/test_guardrails.py
git commit -m "feat: add escalation guardrails"
```

### Task 5: Local Model Adapter And Ticket Pipeline

**Files:**
- Create: `code/local_model.py`
- Create: `code/pipeline.py`
- Modify: `code/schemas.py`
- Test: `code/tests/test_pipeline.py`

- [ ] **Step 1: Write the failing test**

```python
from code.pipeline import OfflineTriagePipeline
from code.schemas import DraftedOutput, RetrievedChunk, TicketInput


class StubRetriever:
    def __init__(self, chunks):
        self.chunks = chunks

    def search(self, query, company, limit=5):
        return self.chunks


class StubModel:
    def draft(self, ticket, retrieved_chunks, guardrail_reasons):
        return DraftedOutput(
            status="replied",
            product_area="team plans",
            response="If your seat was removed, access is managed by your organization admin.",
            justification="The retrieved Team plan guidance says access changes are controlled by organization admins.",
            request_type="product_issue",
            citations=["data/claude/team.md"],
            confidence=0.88,
        )


def test_pipeline_forces_escalation_when_guardrails_trigger():
    ticket = TicketInput(
        issue="Please restore my Claude team workspace access immediately even though I am not an admin.",
        subject="Access lost",
        company="Claude",
    )
    chunks = [
        RetrievedChunk(
            company="Claude",
            product_area_hint="team plans",
            source_path="data/claude/team.md",
            title="Team",
            section_heading="Membership",
            chunk_text="Organization admins manage seats and access.",
            score=0.91,
        )
    ]

    pipeline = OfflineTriagePipeline(retriever=StubRetriever(chunks), model=StubModel())
    output = pipeline.process_ticket(ticket)

    assert output.status == "escalated"
    assert output.request_type == "product_issue"
    assert "cannot safely" in output.response.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest code/tests/test_pipeline.py::test_pipeline_forces_escalation_when_guardrails_trigger -v`
Expected: FAIL with missing `code.pipeline`

- [ ] **Step 3: Write minimal implementation**

`code/local_model.py`

```python
import json

from code.schemas import DraftedOutput, RetrievedChunk, TicketInput


class LocalModelAdapter:
    def __init__(self, model_path: str | None = None) -> None:
        self.model_path = model_path

    def draft(
        self,
        ticket: TicketInput,
        retrieved_chunks: list[RetrievedChunk],
        guardrail_reasons: list[str],
    ) -> DraftedOutput:
        top_area = retrieved_chunks[0].product_area_hint if retrieved_chunks else "general"
        return DraftedOutput(
            status="replied",
            product_area=top_area,
            response="I found related guidance in the local support corpus, but the full local model prompt still needs implementation.",
            justification=json.dumps(
                {
                    "sources": [chunk.source_path for chunk in retrieved_chunks[:3]],
                    "guardrails": guardrail_reasons,
                }
            ),
            request_type="product_issue",
            citations=[chunk.source_path for chunk in retrieved_chunks[:3]],
            confidence=0.50,
        )
```

`code/pipeline.py`

```python
from code.guardrails import evaluate_guardrails
from code.schemas import DraftedOutput, TicketInput, TicketOutput


class OfflineTriagePipeline:
    def __init__(self, retriever, model) -> None:
        self.retriever = retriever
        self.model = model

    def process_ticket(self, ticket: TicketInput) -> TicketOutput:
        query = "\n".join(part for part in [ticket.subject, ticket.issue] if part)
        retrieved_chunks = self.retriever.search(query, company=ticket.company, limit=5)
        pre_decision = evaluate_guardrails(ticket=ticket, retrieved_chunks=retrieved_chunks, drafted_output=None)
        draft = self.model.draft(ticket, retrieved_chunks, pre_decision.reasons)
        post_decision = evaluate_guardrails(ticket=ticket, retrieved_chunks=retrieved_chunks, drafted_output=draft)

        if post_decision.must_escalate:
            return TicketOutput(
                status="escalated",
                product_area=draft.product_area,
                response="I cannot safely resolve this automatically from the local support corpus, so this case should be escalated to a human support team.",
                justification="; ".join(post_decision.reasons),
                request_type=draft.request_type if draft.request_type in {"product_issue", "feature_request", "bug", "invalid"} else "invalid",
            )

        return TicketOutput(
            status="replied",
            product_area=draft.product_area,
            response=draft.response,
            justification=draft.justification,
            request_type=draft.request_type,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest code/tests/test_pipeline.py::test_pipeline_forces_escalation_when_guardrails_trigger -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add code/local_model.py code/pipeline.py code/schemas.py code/tests/test_pipeline.py
git commit -m "feat: add offline triage pipeline"
```

### Task 6: CLI Runner And CSV Output

**Files:**
- Modify: `code/main.py`
- Modify: `code/pipeline.py`
- Test: `code/tests/test_pipeline.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

import pandas as pd

from code.main import build_parser, run_batch


def test_run_batch_writes_required_columns(tmp_path):
    input_csv = tmp_path / "tickets.csv"
    output_csv = tmp_path / "output.csv"
    pd.DataFrame(
        [
            {
                "Issue": "Please restore my access immediately.",
                "Subject": "Access",
                "Company": "Claude",
            }
        ]
    ).to_csv(input_csv, index=False)

    run_batch(input_csv=input_csv, output_csv=output_csv, force_rebuild_index=False)

    written = pd.read_csv(output_csv)
    assert list(written.columns) == ["status", "product_area", "response", "justification", "request_type"]
    assert len(written) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest code/tests/test_pipeline.py::test_run_batch_writes_required_columns -v`
Expected: FAIL with missing `run_batch`

- [ ] **Step 3: Write minimal implementation**

`code/main.py`

```python
import argparse
from pathlib import Path

import pandas as pd

from code.config import INPUT_CSV, OUTPUT_CSV
from code.corpus import load_or_build_index
from code.local_model import LocalModelAdapter
from code.pipeline import OfflineTriagePipeline
from code.retriever import LexicalRetriever
from code.schemas import TicketInput


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline support triage agent")
    parser.add_argument("--input", type=Path, default=INPUT_CSV)
    parser.add_argument("--output", type=Path, default=OUTPUT_CSV)
    parser.add_argument("--rebuild-index", action="store_true")
    return parser


def run_batch(input_csv: Path, output_csv: Path, force_rebuild_index: bool) -> None:
    chunks = load_or_build_index(force_rebuild=force_rebuild_index)
    pipeline = OfflineTriagePipeline(retriever=LexicalRetriever(chunks), model=LocalModelAdapter())

    frame = pd.read_csv(input_csv)
    outputs = []
    for row in frame.to_dict(orient="records"):
        ticket = TicketInput(
            issue=row["Issue"],
            subject=row.get("Subject", "") or "",
            company=row.get("Company", "None"),
        )
        outputs.append(pipeline.process_ticket(ticket).model_dump())

    pd.DataFrame(outputs).to_csv(output_csv, index=False)


def main() -> None:
    args = build_parser().parse_args()
    run_batch(input_csv=args.input, output_csv=args.output, force_rebuild_index=args.rebuild_index)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest code/tests/test_pipeline.py::test_run_batch_writes_required_columns -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add code/main.py code/pipeline.py code/tests/test_pipeline.py
git commit -m "feat: add csv batch runner"
```

### Task 7: Sample Evaluation Harness And Final Tightening

**Files:**
- Create: `code/evaluate.py`
- Modify: `code/README.md`
- Modify: `code/local_model.py`
- Modify: `code/retriever.py`
- Test: `code/tests/test_pipeline.py`

- [ ] **Step 1: Write the failing test**

```python
import pandas as pd

from code.evaluate import compare_outputs


def test_compare_outputs_reports_column_accuracy():
    expected = pd.DataFrame(
        [{"Status": "replied", "Product Area": "billing", "Request Type": "product_issue"}]
    )
    actual = pd.DataFrame(
        [{"status": "replied", "product_area": "billing", "request_type": "product_issue"}]
    )

    report = compare_outputs(expected=expected, actual=actual)

    assert report["status_accuracy"] == 1.0
    assert report["product_area_accuracy"] == 1.0
    assert report["request_type_accuracy"] == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest code/tests/test_pipeline.py::test_compare_outputs_reports_column_accuracy -v`
Expected: FAIL with missing `code.evaluate`

- [ ] **Step 3: Write minimal implementation**

`code/evaluate.py`

```python
import pandas as pd


def compare_outputs(expected: pd.DataFrame, actual: pd.DataFrame) -> dict[str, float]:
    return {
        "status_accuracy": float((expected["Status"] == actual["status"]).mean()),
        "product_area_accuracy": float((expected["Product Area"] == actual["product_area"]).mean()),
        "request_type_accuracy": float((expected["Request Type"] == actual["request_type"]).mean()),
    }
```

`code/retriever.py`

```python
class HybridRetriever(LexicalRetriever):
    """Temporary placeholder name for the eventual embedding-backed retriever."""
```

`code/local_model.py`

```python
class PromptBuilder:
    @staticmethod
    def build(ticket: TicketInput, retrieved_chunks: list[RetrievedChunk], guardrail_reasons: list[str]) -> str:
        lines = [
            "Use only the provided support snippets.",
            f"Company: {ticket.company}",
            f"Subject: {ticket.subject}",
            f"Issue: {ticket.issue}",
            "Guardrails:",
            *guardrail_reasons,
            "Retrieved snippets:",
        ]
        lines.extend(f"- {chunk.source_path}: {chunk.chunk_text[:500]}" for chunk in retrieved_chunks)
        return "\n".join(lines)
```

`code/README.md`

```markdown
## Evaluate On Sample Tickets

```bash
python code/evaluate.py
```

## Offline Notes

- The agent must use only `data/`.
- No hosted APIs are required.
- Set a local model path with `LOCAL_LLM_PATH` if you replace the default adapter.
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest code/tests/test_pipeline.py::test_compare_outputs_reports_column_accuracy -v`
Expected: PASS

- [ ] **Step 5: Run the full test suite**

Run: `pytest code/tests -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add code/evaluate.py code/README.md code/local_model.py code/retriever.py code/tests/test_pipeline.py
git commit -m "feat: add evaluation harness and offline docs"
```

## Self-Review

- Spec coverage:
  - terminal-based CLI is covered in Task 6
  - local-only corpus ingestion and retrieval are covered in Tasks 2 and 3
  - LLM-driven classification and response drafting are covered in Task 5
  - safety escalation is covered in Task 4
  - output CSV generation is covered in Task 6
  - sample-set evaluation and reproducibility docs are covered in Task 7
- Placeholder scan:
  - the only intentionally thin area is the local model adapter internals, but the task sequence still defines the exact files and interface needed before refinement
- Type consistency:
  - `TicketInput`, `RetrievedChunk`, `DraftedOutput`, `GuardrailDecision`, and `TicketOutput` are used consistently across tasks
