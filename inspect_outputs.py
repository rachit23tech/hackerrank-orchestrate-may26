import sys
sys.path.insert(0, 'code')

import pandas as pd
from corpus import load_or_build_index
from retriever import HybridRetriever
from schemas import TicketInput

# Load corpus and retriever
chunks = load_or_build_index()
retriever = HybridRetriever(chunks)

# Load tickets and outputs
tickets_df = pd.read_csv('support_issues/support_issues.csv')
outputs_df = pd.read_csv('support_issues/output.csv')

# Inspect first 5
for i in range(min(5, len(tickets_df))):
    row = tickets_df.iloc[i]
    output = outputs_df.iloc[i]
    
    ticket = TicketInput(
        issue=str(row['Issue']) if pd.notna(row['Issue']) else "",
        subject=str(row['Subject']) if 'Subject' in row and pd.notna(row['Subject']) else "",
        company=str(row['Company']) if 'Company' in row and pd.notna(row['Company']) else "None"
    )
    
    query = "\n".join(part for part in [ticket.subject, ticket.issue] if part)
    retrieved = retriever.search(query, company=ticket.company, limit=3)
    
    print(f"\n{'='*70}")
    print(f"TICKET {i+1}: {ticket.company} - {ticket.subject[:50]}")
    print(f"{'='*70}")
    print(f"Decision: {output['status']} | Type: {output['request_type']} | Area: {output['product_area']}")
    print(f"\nRetrieved {len(retrieved)} chunks:\n")
    
    for j, chunk in enumerate(retrieved, 1):
        print(f"  [{j}] {chunk.source_path}")
        print(f"      Section: {chunk.section_heading}")
        print(f"      Preview: {chunk.chunk_text[:120]}...")
        print()
    
    print(f"Response snippet: {output['response'][:150]}...")
    print(f"Justification: {output['justification']}")

print("\n" + "="*70)
print("MANUAL REVIEW CHECKLIST:")
print("="*70)
print("✓ Are responses grounded in retrieved chunks?")
print("✓ Do escalations make sense given the guardrail rules?")
print("✓ Are product areas correct?")
print("✓ Would you as a support agent be happy with these?")
