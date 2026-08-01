"""Agentic core.
A tool-using agent that decides, per turn, whether it needs to search the knowledge base, look up an order, or
escalate to a human (instead of a single blind LLM call).
"""
from __future__ import annotations

import json
from typing import Callable

from app.llm_backends import call_llm
from app.rag import retrieve

SYSTEM_PROMPT = """You are a customer support agent for an online store.

Rules:
- Always ground factual answers (policies, how-to, product info) in the
  knowledge base via `search_knowledge_base`. Never invent policy details.
- If asked about a specific order, use `get_order_status`.
- If you are not confident you can resolve the issue (angry customer,
  refund disputes, anything outside the knowledge base, or the customer
  explicitly asks for a human), call `escalate_to_human` and tell the
  customer a human will follow up.
- Keep replies concise (2-4 sentences) and friendly.
- Cite which knowledge base source you used in your own reasoning, but
  don't show raw source filenames to the customer.
"""

TOOLS = [
    {
        "name": "search_knowledge_base",
        "description": "Search the support knowledge base (FAQs, policies, guides) for relevant information.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Search query"}},
            "required": ["query"],
        },
    },
    {
        "name": "get_order_status",
        "description": "Look up the status of a customer order by order ID.",
        "input_schema": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
        },
    },
    {
        "name": "escalate_to_human",
        "description": "Hand the conversation off to a human agent. Use when you cannot confidently resolve the issue.",
        "input_schema": {
            "type": "object",
            "properties": {"reason": {"type": "string", "description": "Why this needs a human"}},
            "required": ["reason"],
        },
    },
]


def _mock_order_status(order_id: str) -> dict:
    """Stand-in for a real orders DB/API. Swap for a real query in prod."""
    fake_db = {
        "1001": {"status": "shipped", "eta": "2026-08-03", "carrier": "DHL"},
        "1002": {"status": "processing", "eta": "2026-08-05"},
    }
    return fake_db.get(order_id, {"status": "not_found"})


def run_agent(
    user_message: str,
    conversation_history: list[dict] | None = None,
    on_escalate: Callable[[str], None] | None = None,
) -> tuple[str, list[dict]]:
    """Run one turn of the agent loop. Returns (final_reply, updated_history).

    conversation_history is a list of {"role", "content"} messages, where
    content is either a plain string or a list of blocks (text/tool_use/
    tool_result) — the same shape regardless of backend, so callers can
    persist multi-turn context per Chatwoot conversation either way.
    """
    messages = (conversation_history or []) + [{"role": "user", "content": user_message}]

    for _ in range(5):  # cap tool-use iterations to avoid infinite loops
        response = call_llm(SYSTEM_PROMPT, messages, TOOLS)

        if response["stop_reason"] != "tool_use":
            final_text = "".join(b["text"] for b in response["content"] if b["type"] == "text")
            messages.append({"role": "assistant", "content": response["content"]})
            return final_text, messages

        # Model wants to call one or more tools
        messages.append({"role": "assistant", "content": response["content"]})
        tool_results = []
        for block in response["content"]:
            if block["type"] != "tool_use":
                continue
            result = _dispatch_tool(block["name"], block["input"], on_escalate)
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block["id"],
                    "content": json.dumps(result),
                }
            )
        messages.append({"role": "user", "content": tool_results})

    return "I'm having trouble processing this — let me get a human to help.", messages


def _dispatch_tool(name: str, tool_input: dict, on_escalate: Callable[[str], None] | None) -> dict:
    if name == "search_knowledge_base":
        hits = retrieve(tool_input["query"])
        return {"results": [{"text": h["text"], "relevance": round(h["score"], 3)} for h in hits]}
    if name == "get_order_status":
        return _mock_order_status(tool_input["order_id"])
    if name == "escalate_to_human":
        if on_escalate:
            on_escalate(tool_input.get("reason", "unspecified"))
        return {"status": "escalated"}
    return {"error": f"unknown tool {name}"}
