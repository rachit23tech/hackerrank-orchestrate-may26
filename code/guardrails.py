import re

from schemas import DraftedOutput, GuardrailDecision, RetrievedChunk, TicketInput


HIGH_RISK_PATTERNS = [
    (
        r"\bfraud\b|\bscam\b|\bunauthorized transaction\b|\bcard used without my permission\b",
        "Fraud or suspicious transaction requires escalation.",
    ),
    (
        r"\brestore my\b.*\baccess\b|\bgrant access\b|\bworkspace owner\b|\bnot an admin\b",
        "Access restoration without verified authority requires escalation.",
    ),
    (
        r"\bincrease my score\b|\brejected me\b|\bmove me forward\b|\bmove me to the next round\b",
        "Score or hiring outcome disputes require escalation.",
    ),
    (
        r"\brefund me (?:today|immediately|now)\b|\bprocess my refund\b|\bissue a refund\b|\bban the (?:seller|user)\b|\bsuspend the account\b",
        "Requested account or enforcement action is outside automated scope.",
    ),
    (r"\bsue you\b|\blawsuit\b|\blegal action\b|\bcontact my lawyer\b", "Legal threats or actions require immediate escalation."),
    (
        r"\bignore previous instructions\b|\bsystem prompt\b|\bhidden policy\b|\bbypass safety\b|\boverride system\b",
        "Prompt injection or policy-extraction attempt requires escalation.",
    ),
]

UNSUPPORTED_ACTION_PATTERNS = [
    r"\bi have (?:already )?updated\b",
    r"\bi updated\b",
    r"\bi have (?:already )?contacted\b",
    r"\bi contacted\b",
    r"\bi have (?:already )?reversed\b",
    r"\bi reversed\b",
    r"\bi have (?:already )?refunded\b",
    r"\bi refunded\b",
    r"\bi have (?:already )?restored\b",
    r"\bi restored\b",
    r"\bi have (?:already )?cancelled\b",
    r"\bi cancelled\b",
    r"\bi have (?:already )?processed\b",
    r"\bi processed\b",
    r"\bi have (?:already )?deleted\b",
    r"\bi deleted\b",
]


def evaluate_guardrails(
    ticket: TicketInput,
    retrieved_chunks: list[RetrievedChunk],
    drafted_output: DraftedOutput | None,
) -> GuardrailDecision:
    reasons: list[str] = []
    haystack = f"{ticket.subject}\n{ticket.issue}".lower()
    is_invalid_reply = bool(drafted_output and drafted_output.request_type == "invalid")

    for pattern, reason in HIGH_RISK_PATTERNS:
        if re.search(pattern, haystack):
            reasons.append(reason)

    if not is_invalid_reply and (not retrieved_chunks or max(chunk.score for chunk in retrieved_chunks) < 0.12):
        reasons.append("Evidence confidence is too weak to support a direct reply.")

    if drafted_output and not is_invalid_reply and drafted_output.confidence < 0.40:
        reasons.append("Model confidence is too low for a direct reply.")

    if drafted_output:
        response_text = drafted_output.response.lower()
        if any(re.search(pattern, response_text) for pattern in UNSUPPORTED_ACTION_PATTERNS):
            reasons.append("Unsupported action claim in drafted response requires escalation.")

    return GuardrailDecision(must_escalate=bool(reasons), reasons=reasons)
