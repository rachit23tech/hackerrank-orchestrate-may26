import re

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    import transformers
    # Suppress harmless unexpected key warnings to keep the terminal output clean
    transformers.logging.set_verbosity_error()
except ImportError:
    SentenceTransformer = None

from schemas import RetrievedChunk


DOMAIN_HINTS = {
    "billing": {"billing", "subscription", "refund", "payment", "paid", "order"},
    "travel_support": {"travel", "traveller", "traveler", "trip", "cheque", "cheques"},
    "privacy": {"privacy", "private", "delete", "data", "crawl", "temporary"},
    "community": {"community", "resume", "certificate", "google", "account"},
    "screen": {"test", "assessment", "candidate", "interview", "submission", "recruiter"},
    "fraud": {"fraud", "stolen", "unauthorized", "scam", "blocked"},
    "teams_management": {"employee", "employees", "team", "member", "members", "user", "users", "remove", "revoke"},
    "education": {"professor", "student", "students", "lti", "canvas", "education", "school", "college"},
}


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def is_metadata_only(chunk: RetrievedChunk) -> bool:
    substantive_lines = 0
    for raw_line in chunk.chunk_text.splitlines():
        line = raw_line.strip()
        lower = line.lower()
        if not line:
            continue
        if lower.startswith(
            ("title:", "title_slug:", "source_url:", "article_slug:", "last_updated", "breadcrumbs:", "final_url:")
        ):
            continue
        if lower.startswith("_last updated:"):
            continue
        if line == "---" or re.fullmatch(r'-\s*"[^"]+"', line):
            continue
        substantive_lines += 1
    return substantive_lines == 0


class LexicalRetriever:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b[a-zA-Z0-9][a-zA-Z0-9_]+\b")
        self.chunk_texts = [self._build_index_text(chunk) for chunk in chunks]
        self.chunk_matrix = self.vectorizer.fit_transform(self.chunk_texts)

    @staticmethod
    def _build_index_text(chunk: RetrievedChunk) -> str:
        return "\n".join(
            [
                chunk.title,
                chunk.section_heading,
                chunk.product_area_hint,
                chunk.source_path,
                chunk.chunk_text,
            ]
        )

    @staticmethod
    def _score_chunk(query_tokens: set[str], chunk: RetrievedChunk) -> float:
        score = 0.0
        path_blob = " ".join([chunk.source_path, chunk.title, chunk.section_heading, chunk.product_area_hint]).lower()

        if is_metadata_only(chunk):
            score -= 0.18
        if chunk.source_path.endswith("index.md"):
            score -= 0.08
        if chunk.section_heading.lower() == "overview":
            score -= 0.03

        for product_area, hints in DOMAIN_HINTS.items():
            if query_tokens & hints and any(hint in path_blob for hint in hints):
                score += 0.09
                if product_area.replace("_", " ") in chunk.product_area_hint.lower():
                    score += 0.03

        if "billing" in path_blob and query_tokens & DOMAIN_HINTS["billing"]:
            score += 0.08
        if "resume" in path_blob and "resume" in query_tokens:
            score += 0.08
        if "certificate" in path_blob and "certificate" in query_tokens:
            score += 0.08
        if "fraud" in path_blob and query_tokens & DOMAIN_HINTS["fraud"]:
            score += 0.08
        if "teams-management" in path_blob and query_tokens & DOMAIN_HINTS["teams_management"]:
            score += 0.22
        if "claude-for-education" in path_blob and query_tokens & DOMAIN_HINTS["education"]:
            score += 0.12
        if "delete-account" in path_blob and query_tokens & DOMAIN_HINTS["teams_management"]:
            score -= 0.20

        return score

    def search(self, query: str, company: str | None, limit: int = 5) -> list[RetrievedChunk]:
        filtered_indices: list[int] = []
        filtered_chunks: list[RetrievedChunk] = []
        for index, chunk in enumerate(self.chunks):
            if company and company != "None" and chunk.company != company:
                continue
            filtered_indices.append(index)
            filtered_chunks.append(chunk)

        if not filtered_chunks:
            return []

        query_vector = self.vectorizer.transform([query])
        cosine_scores = (self.chunk_matrix[filtered_indices] @ query_vector.T).toarray().ravel()
        query_tokens = set(tokenize(query))

        ranked: list[tuple[float, RetrievedChunk]] = []
        for local_index, chunk in enumerate(filtered_chunks):
            score = float(cosine_scores[local_index]) + self._score_chunk(query_tokens, chunk)
            ranked.append((score, chunk.model_copy(update={"score": score})))

        ranked.sort(key=lambda item: item[0], reverse=True)
        return [chunk for score, chunk in ranked[:limit] if score > 0.0]


class HybridRetriever(LexicalRetriever):
    """Offline Hybrid Retriever combining TF-IDF, Semantic Embeddings, and Heuristics."""
    
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        super().__init__(chunks)
        self.model = None
        self.embeddings = None
        if SentenceTransformer:
            try:
                # Using a tiny, fast model specifically designed for local offline search
                self.model = SentenceTransformer("all-MiniLM-L6-v2")
                self.embeddings = self.model.encode(self.chunk_texts, convert_to_numpy=True)
            except Exception as e:
                print(f"Warning: Could not load sentence-transformers. Falling back to Lexical-only. {e}")
                self.model = None

    def search(self, query: str, company: str | None, limit: int = 5) -> list[RetrievedChunk]:
        filtered_indices: list[int] = []
        filtered_chunks: list[RetrievedChunk] = []
        for index, chunk in enumerate(self.chunks):
            if company and company != "None" and chunk.company != company:
                continue
            filtered_indices.append(index)
            filtered_chunks.append(chunk)

        if not filtered_chunks:
            return []

        query_vector = self.vectorizer.transform([query])
        lexical_scores = (self.chunk_matrix[filtered_indices] @ query_vector.T).toarray().ravel()
        query_tokens = set(tokenize(query))

        semantic_scores = np.zeros(len(filtered_chunks))
        if self.model is not None and self.embeddings is not None:
            query_embedding = self.model.encode([query], convert_to_numpy=True)[0]
            chunk_embeddings = self.embeddings[filtered_indices]
            norms = np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(query_embedding)
            norms[norms == 0] = 1e-10
            semantic_scores = np.dot(chunk_embeddings, query_embedding) / norms

        ranked: list[tuple[float, RetrievedChunk]] = []
        for local_index, chunk in enumerate(filtered_chunks):
            lex_score = float(lexical_scores[local_index])
            sem_score = float(semantic_scores[local_index])
            heuristic = self._score_chunk(query_tokens, chunk)
            
            # The Hybrid Weighting (40% TF-IDF, 60% Semantic, plus Domain Heuristics)
            final_score = (lex_score * 0.4) + (sem_score * 0.6) + heuristic
            ranked.append((final_score, chunk.model_copy(update={"score": final_score})))

        ranked.sort(key=lambda item: item[0], reverse=True)
        return [chunk for score, chunk in ranked[:limit] if score > 0.0]
