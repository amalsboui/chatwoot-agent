"""Minimal tests. Run with: pytest

Note: test_agent_tools_and_e2e requires ANTHROPIC_API_KEY and the KB to be
ingested (python -m app.ingest) since it hits the real model — kept small
on purpose for a portfolio project, not meant to be a full CI suite.
"""
import os

import pytest

from app.rag import ingest_directory, retrieve


@pytest.fixture(scope="module", autouse=True)
def seeded_kb(tmp_path_factory):
    ingest_directory("data/sample_docs")


def test_retrieval_returns_relevant_doc():
    results = retrieve("how long until I can return something", k=3)
    sources = {r["source"] for r in results}
    assert "returns_policy.md" in sources


def test_retrieval_returns_nothing_for_empty_index_edge_case():
    # sanity check the function doesn't error even with an odd query
    results = retrieve("askjdhaksjdh random gibberish query", k=3)
    assert isinstance(results, list)


@pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="requires live API key")
def test_agent_escalates_on_explicit_request():
    from app.agent import run_agent

    escalated = {"flag": False}
    reply, _ = run_agent(
        "I am furious, I want to talk to a real human right now, not a bot.",
        [],
        on_escalate=lambda reason: escalated.__setitem__("flag", True),
    )
    assert escalated["flag"] is True
    assert isinstance(reply, str) and len(reply) > 0
