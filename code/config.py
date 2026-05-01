from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
SUPPORT_ISSUES_DIR = REPO_ROOT / "support_issues"
INPUT_CSV = SUPPORT_ISSUES_DIR / "support_issues.csv"
SAMPLE_CSV = SUPPORT_ISSUES_DIR / "sample_support_issues.csv"
OUTPUT_CSV = SUPPORT_ISSUES_DIR / "output.csv"
CACHE_DIR = REPO_ROOT / "code" / ".cache"
INDEX_CACHE_PATH = CACHE_DIR / "corpus_index.json"
EMBEDDING_CACHE_PATH = CACHE_DIR / "corpus_embeddings.npy"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_LOCAL_MODEL_PATH_ENV = "LOCAL_LLM_PATH"
