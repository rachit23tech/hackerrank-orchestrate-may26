import argparse
from pathlib import Path

import pandas as pd

from config import INPUT_CSV, OUTPUT_CSV, REPO_ROOT
from corpus import load_or_build_index
from local_model import LocalModelAdapter
from pipeline import OfflineTriagePipeline
from retriever import HybridRetriever
from schemas import TicketInput


FORMULA_PREFIXES = ("=", "+", "-", "@")


def normalize_optional_text(value: object, default: str = "", max_length: int = 4000) -> str:
    if pd.isna(value):
        return default
    return str(value)[:max_length]


def sanitize_csv_cell(value: object) -> str:
    text = normalize_optional_text(value, default="")
    if text.startswith(FORMULA_PREFIXES):
        return f"'{text}"
    return text


def validate_repo_path(path: Path | str, must_exist: bool) -> Path:
    candidate = Path(path).expanduser()
    resolved = candidate.resolve(strict=False)
    repo_root = REPO_ROOT.resolve()
    try:
        resolved.relative_to(repo_root)
    except ValueError as exc:
        raise ValueError("Input and output paths must stay inside the repository.") from exc
    if must_exist and not resolved.exists():
        raise ValueError(f"Required path does not exist: {resolved}")
    return resolved


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline support triage agent")
    parser.add_argument("--input", type=Path, default=INPUT_CSV)
    parser.add_argument("--output", type=Path, default=OUTPUT_CSV)
    parser.add_argument("--rebuild-index", action="store_true")
    return parser


def run_batch(input_csv: Path, output_csv: Path, force_rebuild_index: bool) -> None:
    input_csv = validate_repo_path(input_csv, must_exist=True)
    output_csv = validate_repo_path(output_csv, must_exist=False)
    chunks = load_or_build_index(force_rebuild=force_rebuild_index)
    pipeline = OfflineTriagePipeline(retriever=HybridRetriever(chunks), model=LocalModelAdapter())

    frame = pd.read_csv(input_csv)
    outputs = []
    for row in frame.to_dict(orient="records"):
        ticket = TicketInput(
            issue=normalize_optional_text(row["Issue"], max_length=4000).strip(),
            subject=normalize_optional_text(row.get("Subject", ""), default="", max_length=512).strip(),
            company=normalize_optional_text(row.get("Company", "None"), default="None", max_length=32).strip() or "None",
        )
        output = pipeline.process_ticket(ticket).model_dump()
        outputs.append({key: sanitize_csv_cell(value) for key, value in output.items()})

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(outputs).to_csv(output_csv, index=False)


def main() -> None:
    args = build_parser().parse_args()
    run_batch(input_csv=args.input, output_csv=args.output, force_rebuild_index=args.rebuild_index)


if __name__ == "__main__":
    main()
