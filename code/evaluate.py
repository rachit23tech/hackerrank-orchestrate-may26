from pathlib import Path

import pandas as pd

from config import OUTPUT_CSV, SAMPLE_CSV
from main import run_batch


def compare_outputs(expected: pd.DataFrame, actual: pd.DataFrame) -> dict[str, float]:
    expected_status = expected["Status"].fillna("").str.lower().str.strip()
    actual_status = actual["status"].fillna("").str.lower().str.strip()
    expected_area = expected["Product Area"].fillna("").str.lower().str.strip()
    actual_area = actual["product_area"].fillna("").str.lower().str.strip()
    expected_type = expected["Request Type"].fillna("").str.lower().str.strip()
    actual_type = actual["request_type"].fillna("").str.lower().str.strip()
    area_mask = expected_area != ""
    product_area_accuracy = (
        float((expected_area[area_mask] == actual_area[area_mask]).mean()) if area_mask.any() else 1.0
    )
    return {
        "status_accuracy": float((expected_status == actual_status).mean()),
        "product_area_accuracy": product_area_accuracy,
        "request_type_accuracy": float((expected_type == actual_type).mean()),
    }


def main() -> None:
    temp_output = OUTPUT_CSV.parent / "sample_output.csv"
    run_batch(input_csv=Path(SAMPLE_CSV), output_csv=temp_output, force_rebuild_index=False)
    expected = pd.read_csv(SAMPLE_CSV)
    actual = pd.read_csv(temp_output)
    report = compare_outputs(expected=expected, actual=actual)
    for key, value in report.items():
        print(f"{key}: {value:.3f}")


if __name__ == "__main__":
    main()
