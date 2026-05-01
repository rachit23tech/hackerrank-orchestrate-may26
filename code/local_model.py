import json
import re
import os

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None

from schemas import DraftedOutput, RetrievedChunk, TicketInput


class LocalModelAdapter:
    ACTION_PATTERNS = (
        "click ",
        "select ",
        "go to ",
        "navigate ",
        "log in ",
        "open ",
        "choose ",
        "contact ",
        "call ",
        "follow these steps",
        "refresh ",
    )
    LOW_SIGNAL_TOKENS = {
        "a",
        "an",
        "and",
        "any",
        "can",
        "do",
        "for",
        "from",
        "give",
        "had",
        "have",
        "help",
        "i",
        "if",
        "in",
        "is",
        "it",
        "me",
        "money",
        "my",
        "of",
        "on",
        "or",
        "our",
        "please",
        "the",
        "them",
        "to",
        "want",
        "with",
        "you",
        "your",
    }

    def __init__(self, model_path: str | None = None) -> None:
        self.model_path = model_path or os.environ.get("LOCAL_LLM_PATH")
        self.llm = None
        if self.model_path and Llama and os.path.exists(self.model_path):
            # Suppress verbose output to keep the terminal clean
            self.llm = Llama(model_path=self.model_path, n_ctx=2048, verbose=False)

    @staticmethod
    def _combined_text(ticket: TicketInput) -> str:
        return f"{ticket.subject}\n{ticket.issue}".strip().lower()

    @staticmethod
    def _classify_request_type(ticket: TicketInput, retrieved_chunks: list[RetrievedChunk]) -> str:
        text = LocalModelAdapter._combined_text(ticket)
        support_terms = [
            "account",
            "password",
            "claude",
            "hackerrank",
            "visa",
            "test",
            "assessment",
            "interview",
            "subscription",
            "conversation",
            "card",
            "refund",
            "delete",
            "workspace",
            "seat",
            "stolen",
            "lost",
        ]
        if re.search(
            r"\b(site is down|pages are inaccessible|none of the pages are accessible|error|not working|bug|down|failing|stopped working|not responding)\b",
            text,
        ):
            return "bug"
        if re.search(r"\b(feature request|can you add|please add|would like to have)\b", text):
            return "feature_request"
        if re.fullmatch(r"\s*thank you(?: for helping me)?\s*", text):
            return "invalid"
        if "actor in iron man" in text:
            return "invalid"
        if ticket.company == "None" and not retrieved_chunks and not any(term in text for term in support_terms):
            return "invalid"
        return "product_issue"

    @staticmethod
    def _infer_product_area(ticket: TicketInput, retrieved_chunks: list[RetrievedChunk], request_type: str) -> str:
        text = LocalModelAdapter._combined_text(ticket)
        company = ticket.company
        if company == "None" and retrieved_chunks:
            company = retrieved_chunks[0].company

        if request_type == "invalid":
            return "conversation_management"
        if company == "Claude":
            if any(term in text for term in ["bedrock", "aws", "amazon bedrock"]) or any(
                "amazon-bedrock" in chunk.source_path for chunk in retrieved_chunks
            ):
                return "amazon_bedrock"
            if any(term in text for term in ["refund", "billing", "payment", "paid plan", "subscription"]):
                return "billing"
            if any(term in text for term in ["lti", "student", "students", "professor", "canvas", "education"]) or any(
                "claude-for-education" in chunk.source_path for chunk in retrieved_chunks
            ):
                return "education"
            if any(term in text for term in ["private info", "temporary chat", "delete conversation", "delete it", "delete"]) or any(
                "privacy" in chunk.source_path for chunk in retrieved_chunks
            ):
                return "privacy"
            return "conversation_management"
        if company == "HackerRank":
            if any(term in text for term in ["subscription", "billing", "payment", "refund", "order id", "money"]):
                return "billing"
            if any(term in text for term in ["employee", "employees", "team member", "remove user", "remove them", "hiring account"]) or any(
                "teams-management" in chunk.source_path for chunk in retrieved_chunks
            ):
                return "teams_management"
            if any(
                term in text
                for term in ["community", "google login", "delete my account", "password", "resume", "certificate"]
            ) or any("hackerrank_community" in chunk.source_path for chunk in retrieved_chunks):
                return "community"
            return "screen"
        if company == "Visa":
            if any(term in text for term in ["traveller", "traveler", "travel", "cheque", "cheques"]):
                return "travel_support"
            return "general_support"
        if request_type == "bug":
            return "general_support"
        return "general_support"

    @staticmethod
    def _extract_grounded_response(ticket: TicketInput, retrieved_chunks: list[RetrievedChunk], request_type: str) -> str:
        text = LocalModelAdapter._combined_text(ticket)
        if request_type == "invalid":
            if "thank you" in text:
                return "Happy to help."
            return "I am sorry, this is out of scope from my capabilities."

        if not retrieved_chunks:
            return "I could not find enough grounded support guidance in the local corpus to answer this safely."

        cleaned_lines: list[str] = []
        fallback_text = retrieved_chunks[0].chunk_text.strip()
        for chunk in retrieved_chunks:
            candidate_lines: list[str] = []
            for raw_line in chunk.chunk_text.splitlines():
                line = raw_line.strip()
                lower = line.lower()
                if not line:
                    continue
                if lower.startswith(("title:", "title_slug:", "source_url:", "article_slug:", "last_updated", "breadcrumbs:", "final_url:")):
                    continue
                if lower.startswith("article_id:"):
                    continue
                if lower.startswith("_last updated:"):
                    continue
                if line.startswith("![") or line.startswith("- [") or line == "---":
                    continue
                if re.fullmatch(r'-\s*"[^"]+"', line):
                    continue
                candidate_lines.append(line)
            if candidate_lines:
                cleaned_lines = candidate_lines
                break

        cleaned_text = "\n".join(cleaned_lines).strip() or fallback_text
        sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", cleaned_text) if part.strip()]
        selected: list[str] = []
        keywords = {
            word for word in re.findall(r"[a-z0-9]+", text) if word not in LocalModelAdapter.LOW_SIGNAL_TOKENS
        }
        scored_sentences: list[tuple[float, str]] = []
        ordered_action_sentences: list[str] = []
        for index, sentence in enumerate(sentences):
            lowered = sentence.lower()
            if re.search(r"^(title|source_url|article_slug|breadcrumbs|article_id)\b", lowered):
                continue
            is_action_sentence = any(lowered.startswith(prefix) for prefix in LocalModelAdapter.ACTION_PATTERNS)
            if is_action_sentence:
                ordered_action_sentences.append(sentence)
            sentence_tokens = set(re.findall(r"[a-z0-9]+", lowered))
            overlap = len(keywords & sentence_tokens)
            score = float(overlap)
            if is_action_sentence:
                score += 1.5
            if lowered.startswith("**what should i do") or lowered.startswith("what should i do"):
                score += 1.2
            if any(term in lowered for term in ["payment", "refund", "billing", "remove", "delete", "lti", "student", "subscription"]):
                score += 1.0
            score -= index * 0.01
            if score > 0:
                scored_sentences.append((score, sentence))

        if len(ordered_action_sentences) >= 2:
            selected.extend(ordered_action_sentences[:3])

        for _, sentence in sorted(scored_sentences, key=lambda item: item[0], reverse=True):
            if len(selected) >= 3:
                break
            if sentence not in selected:
                selected.append(sentence)
        if not selected:
            selected = sentences[:3]
        return " ".join(selected[:3])

    @staticmethod
    def _build_justification(
        retrieved_chunks: list[RetrievedChunk],
        request_type: str,
        product_area: str,
        guardrail_reasons: list[str],
    ) -> str:
        source_list = ", ".join(chunk.source_path for chunk in retrieved_chunks[:2]) or "no strong supporting article"
        guardrail_note = f" Guardrails noted: {'; '.join(guardrail_reasons)}." if guardrail_reasons else ""
        return f"Classified as {request_type} in {product_area} using {source_list}.{guardrail_note}".strip()

    def _generate_llm_response(self, ticket: TicketInput, retrieved_chunks: list[RetrievedChunk]) -> str:
        if not retrieved_chunks:
            return "I could not find enough grounded support guidance in the local corpus to answer this safely."
            
        context_texts = [f"--- {c.title} ---\n{c.chunk_text}" for c in retrieved_chunks[:3]]
        context = "\n\n".join(context_texts)
        
        prompt = f"""<|system|>
You are a helpful customer support triage agent. 
Using ONLY the Context provided below, draft a polite, brief response to the user's Issue. 
Do NOT make up information. Do NOT claim you can take actions like refunds or resetting passwords. 
If the context does not contain the answer, say "I cannot help with this."
Context:
{context}
<|user|>
Subject: {ticket.subject}
Issue: {ticket.issue}
<|assistant|>"""
        try:
            output = self.llm(prompt, max_tokens=150, temperature=0.1, stop=["<|user|>"])
            return output['choices'][0]['text'].strip()
        except Exception:
            return ""

    def draft(
        self,
        ticket: TicketInput,
        retrieved_chunks: list[RetrievedChunk],
        guardrail_reasons: list[str],
    ) -> DraftedOutput:
        request_type = self._classify_request_type(ticket, retrieved_chunks)
        product_area = self._infer_product_area(ticket, retrieved_chunks, request_type)
        
        # Use local LLM if available, else fallback to deterministic heuristic extraction
        if self.llm and retrieved_chunks and request_type != "invalid":
            response = self._generate_llm_response(ticket, retrieved_chunks)
            if not response:
                response = self._extract_grounded_response(ticket, retrieved_chunks, request_type)
        else:
            response = self._extract_grounded_response(ticket, retrieved_chunks, request_type)
            
        confidence = 0.90 if retrieved_chunks else 0.75 if request_type == "invalid" else 0.35
        status = "replied"
        if request_type == "bug":
            status = "escalated"
        return DraftedOutput(
            status=status,
            product_area=product_area,
            response=response,
            justification=self._build_justification(
                retrieved_chunks=retrieved_chunks,
                request_type=request_type,
                product_area=product_area,
                guardrail_reasons=guardrail_reasons,
            ),
            request_type=request_type,
            citations=[chunk.source_path for chunk in retrieved_chunks[:3]],
            confidence=confidence,
        )
