"""Three interchangeable LLM backends behind one interface, so the agent
loop in agent.py doesn't care which one is running.
"""
from __future__ import annotations

import json
import time
import uuid

import httpx

from app.config import settings


def _anthropic_call(system: str, messages: list[dict], tools: list[dict]) -> dict:
    import anthropic

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    response = client.messages.create(
        model=settings.agent_model,
        max_tokens=1024,
        system=system,
        tools=tools,
        messages=messages,
    )
    content = []
    for block in response.content:
        if block.type == "text":
            content.append({"type": "text", "text": block.text})
        elif block.type == "tool_use":
            content.append({"type": "tool_use", "id": block.id, "name": block.name, "input": block.input})
    return {"stop_reason": response.stop_reason, "content": content}


def _tools_to_openai_format(tools: list[dict]) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            },
        }
        for t in tools
    ]


def _messages_to_ollama_format(system: str, messages: list[dict]) -> list[dict]:
    out = [{"role": "system", "content": system}]
    for m in messages:
        content = m["content"]
        if isinstance(content, str):
            out.append({"role": m["role"], "content": content})
            continue
        if m["role"] == "assistant":
            text_parts = [b["text"] for b in content if b["type"] == "text"]
            tool_uses = [b for b in content if b["type"] == "tool_use"]
            msg = {"role": "assistant", "content": " ".join(text_parts)}
            if tool_uses:
                msg["tool_calls"] = [
                    {"function": {"name": b["name"], "arguments": b["input"]}} for b in tool_uses
                ]
            out.append(msg)
        else:  # user message carrying tool_result blocks
            for b in content:
                if b["type"] == "tool_result":
                    out.append({"role": "tool", "content": b["content"]})
    return out


def _ollama_call(system: str, messages: list[dict], tools: list[dict]) -> dict:
    payload = {
        "model": settings.ollama_model,
        "messages": _messages_to_ollama_format(system, messages),
        "tools": _tools_to_openai_format(tools),
        "stream": False,
    }
    with httpx.Client(timeout=120) as client:
        r = client.post(f"{settings.ollama_base_url}/api/chat", json=payload)
        r.raise_for_status()
        data = r.json()

    message = data.get("message", {})
    tool_calls = message.get("tool_calls") or []
    content = []
    text = message.get("content", "")
    if text:
        content.append({"type": "text", "text": text})
    for tc in tool_calls:
        fn = tc.get("function", {})
        content.append(
            {
                "type": "tool_use",
                "id": str(uuid.uuid4()),
                "name": fn.get("name"),
                "input": fn.get("arguments", {}),
            }
        )
    stop_reason = "tool_use" if tool_calls else "end"
    return {"stop_reason": stop_reason, "content": content}


def _messages_to_openai_format(system: str, messages: list[dict]) -> list[dict]:
    out = [{"role": "system", "content": system}]
    for m in messages:
        content = m["content"]
        if isinstance(content, str):
            out.append({"role": m["role"], "content": content})
            continue
        if m["role"] == "assistant":
            text_parts = [b["text"] for b in content if b["type"] == "text"]
            tool_uses = [b for b in content if b["type"] == "tool_use"]
            msg = {"role": "assistant", "content": " ".join(text_parts) or None}
            if tool_uses:
                msg["tool_calls"] = [
                    {
                        "id": b["id"],
                        "type": "function",
                        "function": {"name": b["name"], "arguments": json.dumps(b["input"])},
                    }
                    for b in tool_uses
                ]
            out.append(msg)
        else:  # user message carrying tool_result blocks -> OpenAI "tool" messages
            for b in content:
                if b["type"] == "tool_result":
                    out.append({"role": "tool", "tool_call_id": b["tool_use_id"], "content": b["content"]})
    return out


def _openai_compatible_call(system: str, messages: list[dict], tools: list[dict]) -> dict:
    payload = {
        "model": settings.openai_compatible_model,
        "messages": _messages_to_openai_format(system, messages),
        "tools": _tools_to_openai_format(tools),
    }
    headers = {"Authorization": f"Bearer {settings.openai_compatible_api_key}"}

    max_attempts = 4
    for attempt in range(max_attempts):
        with httpx.Client(timeout=60) as client:
            r = client.post(
                f"{settings.openai_compatible_base_url}/chat/completions", json=payload, headers=headers
            )

        if r.status_code == 429 and attempt < max_attempts - 1:
            wait = float(r.headers.get("retry-after", 2 ** (attempt + 1)))
            time.sleep(wait)
            continue
        if r.status_code == 400 and "tool_use_failed" in r.text and attempt < max_attempts - 1:
            continue
        r.raise_for_status()
        data = r.json()
        break

    message = data["choices"][0]["message"]
    tool_calls = message.get("tool_calls") or []
    content = []
    if message.get("content"):
        content.append({"type": "text", "text": message["content"]})
    for tc in tool_calls:
        fn = tc["function"]
        args = fn["arguments"]
        if isinstance(args, str):
            args = json.loads(args) if args else {}
        content.append({"type": "tool_use", "id": tc["id"], "name": fn["name"], "input": args})
    stop_reason = "tool_use" if tool_calls else "end"
    return {"stop_reason": stop_reason, "content": content}

def call_llm(system: str, messages: list[dict], tools: list[dict]) -> dict:
    if settings.llm_provider == "ollama":
        return _ollama_call(system, messages, tools)
    if settings.llm_provider == "openai_compatible":
        return _openai_compatible_call(system, messages, tools)
    return _anthropic_call(system, messages, tools)