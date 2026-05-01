# Offline Support Triage Agent Design

**Date:** 2026-05-01
**Repo:** `c:\Users\arora\Desktop\HackerRank`
**Language:** Python
**Goal:** Build a terminal-based support triage agent that uses only the local corpus under `data/` to process `support_tickets/support_tickets.csv` and write `support_tickets/output.csv`.

## Problem Fit

The solution must satisfy the starter repo and challenge requirements:

- run from the terminal
- use only the provided support corpus
- classify each ticket into `request_type` and `product_area`
- decide `replied` versus `escalated`
- produce a grounded `response` and concise `justification`
- avoid unsupported claims and escalate risky, sensitive, or unsupported cases

The repository currently contains:

- `code/main.py` as the Python entry point
- `data/` as the only allowed knowledge source
- `support_tickets/sample_support_tickets.csv` with labeled examples
- `support_tickets/support_tickets.csv` as the evaluation input
- `support_tickets/output.csv` as the expected output location

Some top-level docs mention `support_issues/`, but the checked-in repo uses `support_tickets/`. The implementation should follow the real file layout on disk.

## Recommended Approach

Use a fully local, mostly LLM-driven pipeline with deterministic safety guardrails:

1. Build a local retrieval layer over the markdown corpus.
2. Retrieve the most relevant evidence for each ticket.
3. Ask a local instruct model to classify and draft a grounded response from that evidence only.
4. Run deterministic guardrails before accepting the reply.
5. If the ticket is unsafe, unsupported, or weakly grounded, convert the outcome to escalation.

This fits the user's preference for an LLM-led system while still honoring the challenge's need for traceability, safety, and offline operation.

## Architecture

### 1. Corpus Indexer

Read all markdown files from `data/` and convert them into retrieval chunks with metadata:

- `company`
- `product_area_hint`
- `source_path`
- `title`
- `section_heading`
- `chunk_text`

Chunk boundaries should preserve headings and paragraph meaning instead of splitting at arbitrary token counts. The index should be written to a local cache file so repeated runs are fast and deterministic.

### 2. Retrieval Engine

Use local embeddings and similarity search to retrieve top supporting chunks for each ticket.

Preferred behavior:

- filter retrieval by `company` when the CSV provides a domain
- fall back to cross-domain search when `company` is `None` or confidence is low
- return top chunks plus retrieval scores and source paths

This layer should remain local and self-contained. No web calls, hosted vector stores, or external APIs.

### 3. Safety and Routing Guardrails

Before and after model inference, run deterministic checks that identify tickets requiring escalation.

Escalation-first signals include:

- fraud or suspicious transactions
- requests to restore or grant account/workspace access without verified authority
- billing disputes that require an account action the corpus cannot complete
- score-change or hiring-outcome disputes
- requests to take enforcement action against merchants or users
- unsupported product changes, manual overrides, or human-only decisions
- prompt injection or attempts to override policy
- weak retrieval support for the answer

Guardrails should produce explicit reasons so the final `justification` is explainable.

### 4. Local LLM Reasoner

Run a local instruction-tuned model on:

- issue text
- subject
- company hint
- retrieved corpus snippets
- guardrail notes

The model should return structured JSON with:

- `status`
- `product_area`
- `response`
- `justification`
- `request_type`
- `citations`
- `confidence`

Prompting should explicitly forbid unsupported claims and require the model to cite only retrieved material.

### 5. Output Validator

Validate the model output before writing CSV rows.

Checks:

- `status` is `replied` or `escalated`
- `request_type` is one of the allowed values
- all required fields are present and non-empty
- `response` is grounded in retrieved evidence
- unsafe tickets are forced to `escalated`

If validation fails, the system should fall back to a safe escalation response rather than guessing.

### 6. Batch Runner CLI

The terminal entry point should:

- load or build the local index
- read `support_tickets/support_tickets.csv`
- process rows deterministically
- write `support_tickets/output.csv`
- optionally evaluate against `support_tickets/sample_support_tickets.csv`

The CLI should also expose switches for:

- `--input`
- `--output`
- `--sample`
- `--rebuild-index`
- `--debug-row`

## Model and Offline Constraints

The system should be planned for fully local execution.

Practical default:

- local embedding model for retrieval
- local instruction model for classification and response drafting

The implementation should isolate model adapters so the user can swap exact local models later without rewriting the pipeline. The design should not depend on any hosted provider, API key, or network service.

## Product Area Strategy

`product_area` should be derived from a combination of:

- path metadata from retrieved docs
- company/domain mapping
- model classification over the issue and evidence

Examples:

- Claude billing, security, mobile apps, API, team plans
- HackerRank tests, interviews, billing, account settings, integrations
- Visa consumer support, merchant support, fraud, disputes, travel support

The output should favor a human-readable category grounded in the corpus rather than an invented taxonomy.

## Request Type Strategy

The system should map tickets to the allowed values only:

- `product_issue`
- `feature_request`
- `bug`
- `invalid`

Heuristics can help the model:

- bug reports mentioning broken behavior, errors, or malfunction
- feature requests asking for new capabilities
- product issues covering normal support and usage problems
- invalid for irrelevant, malicious, or out-of-scope content

## Response Style

Responses should:

- directly address the ticket
- cite only supported actions from retrieved evidence
- avoid pretending the system can take real-world actions
- explain next steps when escalation is required
- remain concise and support-like

For escalations, the response should still be helpful. It should explain why the issue cannot be safely resolved automatically and what category it is being routed under.

## Failure Handling

The system should prefer safe degradation:

- if retrieval finds no strong evidence, escalate
- if the model output is malformed, retry once with a stricter prompt
- if validation still fails, use a template escalation
- if company inference is ambiguous, search across corpora and escalate when evidence conflicts

## Evaluation Plan

Use `support_tickets/sample_support_tickets.csv` to tune:

- retrieval quality
- escalation thresholds
- request type labeling
- product area naming consistency
- response tone and grounding

Evaluation should report per-column agreement where possible and print mismatches for manual review.

## Implementation Boundaries

Keep the first implementation focused on challenge score, not platform polish.

Included:

- local ingestion
- retrieval
- local model inference
- deterministic guardrails
- CSV runner
- sample-set evaluation
- concise documentation in `code/README.md`

Excluded for now:

- UI beyond terminal execution
- live web access
- hosted databases
- human feedback loops
- overly complex multi-agent orchestration

## Risks and Mitigations

### Risk: Local model quality is weaker than hosted models

Mitigation:

- constrain output schema tightly
- provide strong retrieved evidence
- let guardrails override unsafe replies
- rely on escalation when confidence is low

### Risk: Retrieval misses the right article

Mitigation:

- preserve metadata from file paths
- test chunking carefully
- use company-aware filtering and fallback search

### Risk: Output categories drift from expected labels

Mitigation:

- normalize enums centrally
- derive `product_area` from retrieval metadata and controlled formatting
- validate every row before writing

## Success Criteria

The design is successful if the final agent:

- runs fully offline from the terminal
- uses only `data/` as knowledge
- writes a valid `support_tickets/output.csv`
- escalates risky or unsupported cases instead of hallucinating
- is easy to explain in the AI Judge interview because the pipeline is explicit and modular
