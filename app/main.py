"""FastAPI service exposing:
  - POST /webhook/chatwoot   -> receives Chatwoot AgentBot events
  - POST /chat               -> plain JSON endpoint for the standalone demo / testing

Conversation memory is kept in-process per conversation_id (a dict).
Good enough for a portfolio project; swap for Redis if this ever needs to
survive restarts or run multi-instance.
"""
from __future__ import annotations

import hashlib
import hmac

from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel

from app.agent import run_agent
from app.chatwoot_client import escalate_to_human, send_message
from app.config import settings

app = FastAPI(title="AI Support Agent")

_conversation_memory: dict[int, list[dict]] = {}


def _verify_signature(raw_body: bytes, signature: str | None) -> bool:
    if not settings.chatwoot_hmac_secret:
        return True  # signature checking disabled until a secret is configured
    if not signature:
        return False
    expected = hmac.new(settings.chatwoot_hmac_secret.encode(), raw_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)


@app.post("/webhook/chatwoot")
async def chatwoot_webhook(request: Request, x_chatwoot_signature: str | None = Header(default=None)):
    raw = await request.body()
    if not _verify_signature(raw, x_chatwoot_signature):
        raise HTTPException(status_code=401, detail="invalid signature")

    payload = await request.json()
    event = payload.get("event")
    message_type = payload.get("message_type")

    if event != "message_created" or message_type != "incoming":
        return {"status": "ignored"}

    conversation_id = payload["conversation"]["id"]
    content = payload.get("content", "")
    history = _conversation_memory.get(conversation_id, [])

    reply, updated_history = run_agent(
        content,
        history,
        on_escalate=lambda reason: escalate_to_human(conversation_id, reason),
    )
    _conversation_memory[conversation_id] = updated_history

    send_message(conversation_id, reply)
    return {"status": "handled"}


class ChatRequest(BaseModel):
    conversation_id: int = 0
    message: str


@app.post("/chat")
def chat(req: ChatRequest):
    """Same agent, no Chatwoot required — used by the CLI demo and for tests."""
    history = _conversation_memory.get(req.conversation_id, [])
    escalated = {"flag": False, "reason": ""}

    def _on_escalate(reason: str):
        escalated["flag"] = True
        escalated["reason"] = reason

    reply, updated_history = run_agent(req.message, history, on_escalate=_on_escalate)
    _conversation_memory[req.conversation_id] = updated_history
    return {"reply": reply, "escalated": escalated["flag"], "escalation_reason": escalated["reason"]}


@app.get("/health")
def health():
    return {"status": "ok"}