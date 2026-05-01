from schemas import TicketInput, TicketOutput


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


def test_retriever_prefers_company_filtered_chunks():
    from retriever import LexicalRetriever
    from schemas import RetrievedChunk

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


class StubRetriever:
    def __init__(self, chunks):
        self.chunks = chunks

    def search(self, query, company, limit=5):
        return self.chunks


class StubModel:
    def draft(self, ticket, retrieved_chunks, guardrail_reasons):
        from schemas import DraftedOutput

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
    from pipeline import OfflineTriagePipeline
    from schemas import RetrievedChunk, TicketInput

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


def test_run_batch_writes_required_columns(tmp_path, monkeypatch):
    import pandas as pd

    import main

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

    monkeypatch.setattr(main, "validate_repo_path", lambda path, must_exist: path.resolve())
    main.run_batch(input_csv=input_csv, output_csv=output_csv, force_rebuild_index=False)

    written = pd.read_csv(output_csv)
    assert list(written.columns) == ["status", "product_area", "response", "justification", "request_type"]
    assert len(written) == 1


def test_compare_outputs_reports_column_accuracy():
    import pandas as pd

    from evaluate import compare_outputs

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


def test_compare_outputs_normalizes_status_case():
    import pandas as pd

    from evaluate import compare_outputs

    expected = pd.DataFrame(
        [{"Status": "Replied", "Product Area": "screen", "Request Type": "product_issue"}]
    )
    actual = pd.DataFrame(
        [{"status": "replied", "product_area": "screen", "request_type": "product_issue"}]
    )

    report = compare_outputs(expected=expected, actual=actual)

    assert report["status_accuracy"] == 1.0


def test_compare_outputs_ignores_blank_expected_product_area():
    import pandas as pd

    from evaluate import compare_outputs

    expected = pd.DataFrame(
        [
            {"Status": "Replied", "Product Area": "", "Request Type": "invalid"},
            {"Status": "Replied", "Product Area": "screen", "Request Type": "product_issue"},
        ]
    )
    actual = pd.DataFrame(
        [
            {"status": "replied", "product_area": "conversation_management", "request_type": "invalid"},
            {"status": "replied", "product_area": "screen", "request_type": "product_issue"},
        ]
    )

    report = compare_outputs(expected=expected, actual=actual)

    assert report["product_area_accuracy"] == 1.0


def test_run_batch_normalizes_blank_csv_cells(tmp_path, monkeypatch):
    import pandas as pd

    import main

    input_csv = tmp_path / "tickets.csv"
    output_csv = tmp_path / "output.csv"
    pd.DataFrame(
        [
            {
                "Issue": "What plan is this workspace on?",
                "Subject": None,
                "Company": None,
            }
        ]
    ).to_csv(input_csv, index=False)

    monkeypatch.setattr(main, "validate_repo_path", lambda path, must_exist: path.resolve())
    main.run_batch(input_csv=input_csv, output_csv=output_csv, force_rebuild_index=False)

    written = pd.read_csv(output_csv)
    assert len(written) == 1
    assert written.loc[0, "status"] in {"replied", "escalated"}


def test_run_batch_escapes_spreadsheet_formula_cells(tmp_path):
    from main import sanitize_csv_cell

    assert sanitize_csv_cell("=CMD|' /C calc'!A0").startswith("'=")
    assert sanitize_csv_cell("@danger").startswith("'@")
    assert sanitize_csv_cell("normal text") == "normal text"


def test_normalize_optional_text_truncates_untrusted_input():
    from main import normalize_optional_text

    value = normalize_optional_text("x" * 6000, max_length=128)

    assert len(value) == 128


def test_validate_repo_path_rejects_outside_paths():
    from main import validate_repo_path

    try:
        validate_repo_path("C:\\Windows\\Temp\\outside.csv", must_exist=False)
    except ValueError as exc:
        assert "inside the repository" in str(exc)
    else:
        raise AssertionError("Expected ValueError for path outside repo")


def test_validate_repo_path_accepts_repo_relative_output(tmp_path, monkeypatch):
    from pathlib import Path
    import main

    monkeypatch.setattr(main, "REPO_ROOT", tmp_path)
    target = tmp_path / "support_tickets" / "output.csv"
    target.parent.mkdir(parents=True, exist_ok=True)

    resolved = main.validate_repo_path(target, must_exist=False)

    assert resolved == target.resolve()


def test_pipeline_normalizes_product_area_and_rejects_unsupported_replies():
    from pipeline import OfflineTriagePipeline
    from schemas import DraftedOutput, RetrievedChunk, TicketInput

    class LocalRetriever:
        def search(self, query, company, limit=5):
            return [
                RetrievedChunk(
                    company="Visa",
                    product_area_hint="travel support",
                    source_path="data/visa/example.md",
                    title="Visa",
                    section_heading="Support",
                    chunk_text="Contact support for travel card issues.",
                    score=0.91,
                )
            ]

    class UnsafeModel:
        def draft(self, ticket, retrieved_chunks, guardrail_reasons):
            return DraftedOutput(
                status="replied",
                product_area="Travel-Support!!!",
                response="I have already reversed the merchant charge for you.",
                justification="Completed the requested action.",
                request_type="product_issue",
                citations=["data/visa/example.md"],
                confidence=0.95,
            )

    ticket = TicketInput(issue="Refund this charge", subject="Chargeback", company="Visa")
    pipeline = OfflineTriagePipeline(retriever=LocalRetriever(), model=UnsafeModel())

    output = pipeline.process_ticket(ticket)

    assert output.status == "escalated"
    assert output.product_area == "travel_support"


def test_pipeline_replies_with_privacy_area_for_claude_delete_conversation():
    from pipeline import OfflineTriagePipeline
    from schemas import DraftedOutput, RetrievedChunk, TicketInput

    class LocalRetriever:
        def search(self, query, company, limit=5):
            return [
                RetrievedChunk(
                    company="Claude",
                    product_area_hint="privacy",
                    source_path="data/claude/privacy/delete-conversation.md",
                    title="Delete conversation",
                    section_heading="Delete",
                    chunk_text="To delete an individual conversation, open it, click the conversation name, and choose Delete.",
                    score=0.95,
                )
            ]

    class HeuristicModel:
        def draft(self, ticket, retrieved_chunks, guardrail_reasons):
            from local_model import LocalModelAdapter

            return LocalModelAdapter().draft(ticket, retrieved_chunks, guardrail_reasons)

    ticket = TicketInput(
        issue="One of my Claude conversations has private info. Can I delete it?",
        subject="",
        company="Claude",
    )
    pipeline = OfflineTriagePipeline(retriever=LocalRetriever(), model=HeuristicModel())

    output = pipeline.process_ticket(ticket)

    assert output.status == "replied"
    assert output.product_area == "privacy"
    assert "delete" in output.response.lower()


def test_pipeline_marks_out_of_scope_question_invalid():
    from pipeline import OfflineTriagePipeline
    from schemas import TicketInput

    class EmptyRetriever:
        def search(self, query, company, limit=5):
            return []

    class HeuristicModel:
        def draft(self, ticket, retrieved_chunks, guardrail_reasons):
            from local_model import LocalModelAdapter

            return LocalModelAdapter().draft(ticket, retrieved_chunks, guardrail_reasons)

    ticket = TicketInput(
        issue="What is the name of the actor in Iron Man?",
        subject="Urgent, please help",
        company="None",
    )
    pipeline = OfflineTriagePipeline(retriever=EmptyRetriever(), model=HeuristicModel())

    output = pipeline.process_ticket(ticket)

    assert output.status == "replied"
    assert output.request_type == "invalid"
    assert output.product_area == "conversation_management"


def test_local_model_skips_frontmatter_and_metadata_in_response():
    from local_model import LocalModelAdapter
    from schemas import RetrievedChunk, TicketInput

    ticket = TicketInput(
        issue="Please pause our subscription for now.",
        subject="Subscription pause",
        company="HackerRank",
    )
    chunks = [
        RetrievedChunk(
            company="HackerRank",
            product_area_hint="settings / user account settings and preferences",
            source_path="data/hackerrank/settings/user-account-settings-and-preferences/5157311476-pause-subscription.md",
            title="Pause Subscription",
            section_heading="How to Pause Your Subscription",
            chunk_text=(
                'title: "Pause Subscription"\n'
                'source_url: "https://support.hackerrank.com/articles/5157311476-pause-subscription"\n'
                "_Last updated: Mar 6, 2026, 12:18 AM (Last updated 2 months ago)_\n"
                "The Pause Subscription feature allows individual self-serve plan subscribers to temporarily pause their HackerRank subscription.\n"
                "Follow these steps to pause your subscription.\n"
                "Click the profile icon in the top-right corner and select Settings.\n"
                "Navigate to the Billing section under Subscription.\n"
                "Click the Cancel Plan button and select the pause duration.\n"
            ),
            score=0.95,
        )
    ]

    draft = LocalModelAdapter().draft(ticket, chunks, [])

    assert "title:" not in draft.response.lower()
    assert "source_url:" not in draft.response.lower()
    assert "last updated" not in draft.response.lower()
    assert "click the profile icon" in draft.response.lower()


def test_local_model_uses_next_substantive_chunk_when_first_chunk_is_metadata_only():
    from local_model import LocalModelAdapter
    from schemas import RetrievedChunk, TicketInput

    ticket = TicketInput(
        issue="Hi, please pause our subscription. We have stopped all hiring efforts for now.",
        subject="Subscription pause",
        company="HackerRank",
    )
    chunks = [
        RetrievedChunk(
            company="HackerRank",
            product_area_hint="settings / user account settings and preferences",
            source_path="data/hackerrank/settings/user-account-settings-and-preferences/5157311476-pause-subscription.md",
            title="Pause Subscription",
            section_heading="overview",
            chunk_text=(
                'title: "Pause Subscription"\n'
                'title_slug: "pause-subscription"\n'
                'source_url: "https://support.hackerrank.com/articles/5157311476-pause-subscription"\n'
                "breadcrumbs:\n"
                '- "Settings"\n'
                '- "User Account Settings and Preferences"\n'
            ),
            score=0.99,
        ),
        RetrievedChunk(
            company="HackerRank",
            product_area_hint="settings / user account settings and preferences",
            source_path="data/hackerrank/settings/user-account-settings-and-preferences/5157311476-pause-subscription.md",
            title="Pause Subscription",
            section_heading="How to Pause Your Subscription",
            chunk_text=(
                "Follow these steps to pause your subscription.\n"
                "Click the profile icon in the top-right corner and select Settings.\n"
                "Navigate to the Billing section under Subscription.\n"
                "Click the Cancel Plan button and select the pause duration.\n"
            ),
            score=0.80,
        ),
    ]

    draft = LocalModelAdapter().draft(ticket, chunks, [])

    assert 'settings' not in draft.response.lower()[:30]
    assert "click the profile icon" in draft.response.lower()


def test_local_model_skips_article_id_metadata_lines():
    from local_model import LocalModelAdapter
    from schemas import RetrievedChunk, TicketInput

    ticket = TicketInput(
        issue="I am a professor and want to set up a Claude LTI key for my students.",
        subject="Claude for students",
        company="Claude",
    )
    chunks = [
        RetrievedChunk(
            company="Claude",
            product_area_hint="claude for education",
            source_path="data/claude/claude-for-education/11725453-set-up-the-claude-lti-in-canvas-by-instructure.md",
            title="Set up the Claude LTI",
            section_heading="Getting started",
            chunk_text=(
                'article_id: "11725453"\n'
                "Set up the Claude LTI integration for students in Canvas by Instructure.\n"
                "Open the admin settings and configure the LTI credentials for your institution.\n"
            ),
            score=0.95,
        )
    ]

    draft = LocalModelAdapter().draft(ticket, chunks, [])

    assert "article_id:" not in draft.response.lower()
    assert "lti" in draft.response.lower()


def test_local_model_prefers_relevant_payment_faq_answer():
    from local_model import LocalModelAdapter
    from schemas import RetrievedChunk, TicketInput

    ticket = TicketInput(
        issue="I had an issue with my payment with order ID: cs_live_abcdefgh. Can you help me?",
        subject="Give me my money",
        company="HackerRank",
    )
    chunks = [
        RetrievedChunk(
            company="HackerRank",
            product_area_hint="community / subscriptions payments and billing",
            source_path="data/hackerrank/hackerrank_community/subscriptions-payments-and-billing/9157064719-payments-and-billing-faqs.md",
            title="Payments and billing FAQs",
            section_heading="overview",
            chunk_text=(
                "**What happens to my mock interview credits if I merge accounts?**\n"
                "Mock interview credits cannot be transferred and stay on the account where they were purchased.\n"
                "**What should I do if my payment fails?**\n"
                "Refresh the page and retry the payment. If any amount was deducted incorrectly, it will be refunded within 5-10 business days.\n"
            ),
            score=0.95,
        )
    ]

    draft = LocalModelAdapter().draft(ticket, chunks, [])

    assert "payment fails" in draft.response.lower() or "retry the payment" in draft.response.lower()
    assert "merge accounts" not in draft.response.lower()


def test_local_model_prefers_team_member_removal_steps():
    from local_model import LocalModelAdapter
    from schemas import RetrievedChunk, TicketInput

    ticket = TicketInput(
        issue="One employee has left and I want to remove them from our HackerRank hiring account.",
        subject="Employee leaving the company",
        company="HackerRank",
    )
    chunks = [
        RetrievedChunk(
            company="HackerRank",
            product_area_hint="settings / teams management",
            source_path="data/hackerrank/settings/teams-management/2203617737-manage-team-members.md",
            title="Manage team members",
            section_heading="Removing a team member",
            chunk_text=(
                "To remove a team member from the team:\n"
                "Locate the user in the list.\n"
                "Select the delete icon in the Action column.\n"
            ),
            score=0.95,
        ),
        RetrievedChunk(
            company="HackerRank",
            product_area_hint="settings / teams management",
            source_path="data/hackerrank/settings/teams-management/9603546665-types-of-user-roles.md",
            title="Types of user roles",
            section_heading="Company Admin",
            chunk_text=(
                "Company admins hold the highest level of access and control within the HackerRank platform.\n"
                "They oversee user management, billing, and settings.\n"
            ),
            score=0.90,
        ),
    ]

    draft = LocalModelAdapter().draft(ticket, chunks, [])

    assert "remove a team member" in draft.response.lower() or "delete icon" in draft.response.lower()
    assert "highest level of access" not in draft.response.lower()


def test_pipeline_routes_hackerrank_subscription_request_to_billing():
    from pipeline import OfflineTriagePipeline
    from schemas import RetrievedChunk, TicketInput

    class LocalRetriever:
        def search(self, query, company, limit=5):
            return [
                RetrievedChunk(
                    company="HackerRank",
                    product_area_hint="settings / user account settings and preferences",
                    source_path="data/hackerrank/settings/user-account-settings-and-preferences/5157311476-pause-subscription.md",
                    title="Pause Subscription",
                    section_heading="How to Pause Your Subscription",
                    chunk_text=(
                        "Follow these steps to pause your subscription.\n"
                        "Click the profile icon in the top-right corner and select Settings.\n"
                        "Navigate to the Billing section under Subscription.\n"
                    ),
                    score=0.9,
                )
            ]

    class HeuristicModel:
        def draft(self, ticket, retrieved_chunks, guardrail_reasons):
            from local_model import LocalModelAdapter

            return LocalModelAdapter().draft(ticket, retrieved_chunks, guardrail_reasons)

    ticket = TicketInput(
        issue="Hi, please pause our subscription. We have stopped all hiring efforts for now.",
        subject="Subscription pause",
        company="HackerRank",
    )
    pipeline = OfflineTriagePipeline(retriever=LocalRetriever(), model=HeuristicModel())

    output = pipeline.process_ticket(ticket)

    assert output.status == "replied"
    assert output.product_area == "billing"


def test_pipeline_routes_hackerrank_employee_removal_to_teams_management():
    from pipeline import OfflineTriagePipeline
    from schemas import RetrievedChunk, TicketInput

    class LocalRetriever:
        def search(self, query, company, limit=5):
            return [
                RetrievedChunk(
                    company="HackerRank",
                    product_area_hint="settings / teams management",
                    source_path="data/hackerrank/settings/teams-management/2203617737-manage-team-members.md",
                    title="Manage team members",
                    section_heading="Remove a user",
                    chunk_text="Admins can manage team members, remove users, and update team access from the team settings page.",
                    score=0.9,
                )
            ]

    class HeuristicModel:
        def draft(self, ticket, retrieved_chunks, guardrail_reasons):
            from local_model import LocalModelAdapter

            return LocalModelAdapter().draft(ticket, retrieved_chunks, guardrail_reasons)

    ticket = TicketInput(
        issue="One employee has left and I want to remove them from our HackerRank hiring account.",
        subject="Employee leaving the company",
        company="HackerRank",
    )
    pipeline = OfflineTriagePipeline(retriever=LocalRetriever(), model=HeuristicModel())

    output = pipeline.process_ticket(ticket)

    assert output.status == "replied"
    assert output.product_area == "teams_management"


def test_pipeline_routes_claude_lti_request_to_education():
    from pipeline import OfflineTriagePipeline
    from schemas import RetrievedChunk, TicketInput

    class LocalRetriever:
        def search(self, query, company, limit=5):
            return [
                RetrievedChunk(
                    company="Claude",
                    product_area_hint="claude for education",
                    source_path="data/claude/claude-for-education/11725453-set-up-the-claude-lti-in-canvas-by-instructure.md",
                    title="Set up the Claude LTI",
                    section_heading="Getting started",
                    chunk_text="Set up the Claude LTI integration for students in Canvas by Instructure.",
                    score=0.9,
                )
            ]

    class HeuristicModel:
        def draft(self, ticket, retrieved_chunks, guardrail_reasons):
            from local_model import LocalModelAdapter

            return LocalModelAdapter().draft(ticket, retrieved_chunks, guardrail_reasons)

    ticket = TicketInput(
        issue="I am a professor and want to set up a Claude LTI key for my students.",
        subject="Claude for students",
        company="Claude",
    )
    pipeline = OfflineTriagePipeline(retriever=LocalRetriever(), model=HeuristicModel())

    output = pipeline.process_ticket(ticket)

    assert output.status == "replied"
    assert output.product_area == "education"


def test_pipeline_replies_with_remove_team_member_steps():
    from pipeline import OfflineTriagePipeline
    from schemas import RetrievedChunk, TicketInput

    class LocalRetriever:
        def search(self, query, company, limit=5):
            return [
                RetrievedChunk(
                    company="HackerRank",
                    product_area_hint="settings / teams management",
                    source_path="data/hackerrank/settings/teams-management/2203617737-manage-team-members.md",
                    title="Manage team members",
                    section_heading="Removing a team member",
                    chunk_text=(
                        "To remove a team member from the team:\n"
                        "Locate the user in the list.\n"
                        "Select the delete icon in the Action column.\n"
                    ),
                    score=0.95,
                )
            ]

    class HeuristicModel:
        def draft(self, ticket, retrieved_chunks, guardrail_reasons):
            from local_model import LocalModelAdapter

            return LocalModelAdapter().draft(ticket, retrieved_chunks, guardrail_reasons)

    ticket = TicketInput(
        issue="Hello! I am trying to remove an interviewer from the platform.",
        subject="How to Remove a User",
        company="HackerRank",
    )
    pipeline = OfflineTriagePipeline(retriever=LocalRetriever(), model=HeuristicModel())

    output = pipeline.process_ticket(ticket)

    assert output.status == "replied"
    assert output.product_area == "teams_management"
    assert "delete icon" in output.response.lower()


def test_pipeline_replies_with_aws_support_guidance_for_bedrock_issue():
    from pipeline import OfflineTriagePipeline
    from schemas import RetrievedChunk, TicketInput

    class LocalRetriever:
        def search(self, query, company, limit=5):
            return [
                RetrievedChunk(
                    company="Claude",
                    product_area_hint="amazon bedrock",
                    source_path="data/claude/amazon-bedrock/7996921-i-use-claude-in-amazon-bedrock-who-do-i-contact-for-customer-support-inquiries.md",
                    title="Claude in Amazon Bedrock support",
                    section_heading="overview",
                    chunk_text=(
                        "Contact AWS Support for Claude in Amazon Bedrock support inquiries or reach out to your AWS account manager.\n"
                        "For community-based support, visit AWS re:Post.\n"
                    ),
                    score=0.95,
                )
            ]

    class HeuristicModel:
        def draft(self, ticket, retrieved_chunks, guardrail_reasons):
            from local_model import LocalModelAdapter

            return LocalModelAdapter().draft(ticket, retrieved_chunks, guardrail_reasons)

    ticket = TicketInput(
        issue="I am facing multiple issues in my project. all requests to claude with aws bedrock is failing",
        subject="Issues in Project",
        company="Claude",
    )
    pipeline = OfflineTriagePipeline(retriever=LocalRetriever(), model=HeuristicModel())

    output = pipeline.process_ticket(ticket)

    assert output.status == "escalated"
    assert output.product_area == "amazon_bedrock"
    assert "aws support" in output.justification.lower() or "amazon_bedrock" == output.product_area


def test_pipeline_escalates_when_service_requests_are_failing():
    from pipeline import OfflineTriagePipeline
    from schemas import TicketInput

    class EmptyRetriever:
        def search(self, query, company, limit=5):
            return []

    class HeuristicModel:
        def draft(self, ticket, retrieved_chunks, guardrail_reasons):
            from local_model import LocalModelAdapter

            return LocalModelAdapter().draft(ticket, retrieved_chunks, guardrail_reasons)

    ticket = TicketInput(
        issue="Claude has stopped working completely, all requests are failing.",
        subject="Claude not responding",
        company="Claude",
    )
    pipeline = OfflineTriagePipeline(retriever=EmptyRetriever(), model=HeuristicModel())

    output = pipeline.process_ticket(ticket)

    assert output.status == "escalated"
    assert output.request_type == "bug"


def test_pipeline_uses_draft_justification_when_model_requests_escalation():
    from pipeline import OfflineTriagePipeline
    from schemas import DraftedOutput, RetrievedChunk, TicketInput

    class StrongRetriever:
        def search(self, query, company, limit=5):
            return [
                RetrievedChunk(
                    company="HackerRank",
                    product_area_hint="screen",
                    source_path="data/hackerrank/example.md",
                    title="Example",
                    section_heading="Outage",
                    chunk_text="General troubleshooting information.",
                    score=0.8,
                )
            ]

    class EscalatingModel:
        def draft(self, ticket, retrieved_chunks, guardrail_reasons):
            return DraftedOutput(
                status="escalated",
                product_area="screen",
                response="Escalate to a human.",
                justification="Classified as a bug with no grounded article to answer from.",
                request_type="bug",
                citations=[],
                confidence=0.9,
            )

    ticket = TicketInput(issue="site is down & none of the pages are accessible", subject="", company="None")
    pipeline = OfflineTriagePipeline(retriever=StrongRetriever(), model=EscalatingModel())

    output = pipeline.process_ticket(ticket)

    assert output.status == "escalated"
    assert "bug" in output.justification.lower()
