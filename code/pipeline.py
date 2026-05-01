from guardrails import evaluate_guardrails
from schemas import TicketInput, TicketOutput
import re


def normalize_product_area(value: str, fallback: str) -> str:
    source = value.strip() or fallback
    normalized = re.sub(r"[^a-z0-9]+", "_", source.lower()).strip("_")
    fallback_normalized = re.sub(r"[^a-z0-9]+", "_", fallback.lower()).strip("_")
    return normalized or fallback_normalized or "general_support"


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
        fallback_area = retrieved_chunks[0].product_area_hint if retrieved_chunks else "general"
        product_area = normalize_product_area(draft.product_area, fallback_area)

        if draft.status == "escalated" or post_decision.must_escalate:
            request_type = draft.request_type
            if request_type not in {"product_issue", "feature_request", "bug", "invalid"}:
                request_type = "invalid"
            justification = "; ".join(post_decision.reasons).strip() or draft.justification
            return TicketOutput(
                status="escalated",
                product_area=product_area,
                response="I cannot safely resolve this automatically from the local support corpus, so this case should be escalated to a human support team.",
                justification=justification,
                request_type=request_type,
            )

        return TicketOutput(
            status="replied",
            product_area=product_area,
            response=draft.response,
            justification=draft.justification,
            request_type=draft.request_type,
        )
