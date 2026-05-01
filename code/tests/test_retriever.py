from retriever import LexicalRetriever
from schemas import RetrievedChunk


def test_retriever_prefers_substantive_chunk_over_metadata_only_chunk():
    retriever = LexicalRetriever(
        chunks=[
            RetrievedChunk(
                company="HackerRank",
                product_area_hint="settings / billing",
                source_path="data/hackerrank/settings/user-account-settings-and-preferences/515-pause-subscription.md",
                title="Pause Subscription",
                section_heading="overview",
                chunk_text=(
                    'title: "Pause Subscription"\n'
                    'source_url: "https://support.hackerrank.com/articles/515-pause-subscription"\n'
                    "breadcrumbs:\n"
                    '- "Settings"\n'
                    '- "User Account Settings and Preferences"\n'
                ),
                score=0.0,
            ),
            RetrievedChunk(
                company="HackerRank",
                product_area_hint="settings / billing",
                source_path="data/hackerrank/settings/user-account-settings-and-preferences/515-pause-subscription.md",
                title="Pause Subscription",
                section_heading="How to Pause Your Subscription",
                chunk_text=(
                    "Follow these steps to pause your subscription.\n"
                    "Click the profile icon in the top-right corner and select Settings.\n"
                    "Navigate to the Billing section under Subscription.\n"
                ),
                score=0.0,
            ),
        ]
    )

    results = retriever.search("please pause our subscription", company="HackerRank", limit=2)

    assert len(results) == 2
    assert results[0].section_heading == "How to Pause Your Subscription"


def test_retriever_boosts_billing_match_for_hackerrank_subscription_queries():
    retriever = LexicalRetriever(
        chunks=[
            RetrievedChunk(
                company="HackerRank",
                product_area_hint="community",
                source_path="data/hackerrank/hackerrank_community/account-settings/manage-account/561-delete-account.md",
                title="Delete account",
                section_heading="Delete",
                chunk_text="To delete your HackerRank Community account, log in and open Settings.",
                score=0.0,
            ),
            RetrievedChunk(
                company="HackerRank",
                product_area_hint="settings / billing",
                source_path="data/hackerrank/settings/user-account-settings-and-preferences/515-pause-subscription.md",
                title="Pause Subscription",
                section_heading="How to Pause Your Subscription",
                chunk_text=(
                    "The Pause Subscription feature allows you to temporarily pause your subscription.\n"
                    "Navigate to the Billing section and choose a pause duration.\n"
                ),
                score=0.0,
            ),
        ]
    )

    results = retriever.search("please pause our subscription", company="HackerRank", limit=2)

    assert results
    assert "pause-subscription" in results[0].source_path


def test_retriever_boosts_team_management_for_employee_removal_queries():
    retriever = LexicalRetriever(
        chunks=[
            RetrievedChunk(
                company="HackerRank",
                product_area_hint="community",
                source_path="data/hackerrank/hackerrank_community/account-settings/manage-account/561-delete-account.md",
                title="Delete account",
                section_heading="Delete",
                chunk_text="To delete your HackerRank Community account, log in and open Settings.",
                score=0.0,
            ),
            RetrievedChunk(
                company="HackerRank",
                product_area_hint="settings / teams management",
                source_path="data/hackerrank/settings/teams-management/2203617737-manage-team-members.md",
                title="Manage team members",
                section_heading="Remove a user",
                chunk_text="Admins can manage team members, remove users, and update team access from the team settings page.",
                score=0.0,
            ),
        ]
    )

    results = retriever.search(
        "one employee has left and I want to remove them from our hiring account",
        company="HackerRank",
        limit=2,
    )

    assert results
    assert "manage-team-members" in results[0].source_path


def test_retriever_boosts_claude_education_for_lti_queries():
    retriever = LexicalRetriever(
        chunks=[
            RetrievedChunk(
                company="Claude",
                product_area_hint="pro and max plans / general",
                source_path="data/claude/pro-and-max-plans/general/12386328-requesting-a-refund-for-a-paid-claude-plan.md",
                title="Requesting a refund",
                section_heading="Refund",
                chunk_text="Request a refund for a paid Claude plan from the support messenger.",
                score=0.0,
            ),
            RetrievedChunk(
                company="Claude",
                product_area_hint="claude for education",
                source_path="data/claude/claude-for-education/11725453-set-up-the-claude-lti-in-canvas-by-instructure.md",
                title="Set up the Claude LTI",
                section_heading="Getting started",
                chunk_text="Set up the Claude LTI integration for students in Canvas by Instructure.",
                score=0.0,
            ),
        ]
    )

    results = retriever.search(
        "I am a professor and want to set up a Claude LTI key for my students",
        company="Claude",
        limit=2,
    )

    assert results
    assert "claude-for-education" in results[0].source_path


def test_retriever_boosts_manage_team_members_for_remove_interviewer_queries():
    retriever = LexicalRetriever(
        chunks=[
            RetrievedChunk(
                company="HackerRank",
                product_area_hint="settings / teams management",
                source_path="data/hackerrank/settings/teams-management/9603546665-types-of-user-roles.md",
                title="Types of user roles",
                section_heading="Company Admin",
                chunk_text="Company admins hold the highest level of access and control within the HackerRank platform.",
                score=0.0,
            ),
            RetrievedChunk(
                company="HackerRank",
                product_area_hint="settings / teams management",
                source_path="data/hackerrank/settings/teams-management/2203617737-manage-team-members.md",
                title="Manage team members",
                section_heading="Removing a team member",
                chunk_text="To remove a team member from the team, locate the user and select the delete icon in the Action column.",
                score=0.0,
            ),
        ]
    )

    results = retriever.search(
        "trying to remove an interviewer from the platform",
        company="HackerRank",
        limit=2,
    )

    assert results
    assert "manage-team-members" in results[0].source_path


def test_retriever_boosts_bedrock_support_article_for_aws_failure_queries():
    retriever = LexicalRetriever(
        chunks=[
            RetrievedChunk(
                company="Claude",
                product_area_hint="claude for education",
                source_path="data/claude/claude-for-education/11725453-set-up-the-claude-lti-in-canvas-by-instructure.md",
                title="Set up the Claude LTI",
                section_heading="Getting started",
                chunk_text="Set up the Claude LTI integration for students in Canvas by Instructure.",
                score=0.0,
            ),
            RetrievedChunk(
                company="Claude",
                product_area_hint="amazon bedrock",
                source_path="data/claude/amazon-bedrock/7996921-i-use-claude-in-amazon-bedrock-who-do-i-contact-for-customer-support-inquiries.md",
                title="Claude in Amazon Bedrock support",
                section_heading="overview",
                chunk_text="Contact AWS Support for Claude in Amazon Bedrock support inquiries or reach out to your AWS account manager.",
                score=0.0,
            ),
        ]
    )

    results = retriever.search(
        "all requests to claude with aws bedrock is failing",
        company="Claude",
        limit=2,
    )

    assert results
    assert "amazon-bedrock" in results[0].source_path
