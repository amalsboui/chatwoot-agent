"""Thin wrapper around the Chatwoot REST API used by the agent to reply
and hand off conversations to a human."""
from __future__ import annotations

import httpx

from app.config import settings


def _headers() -> dict:
    return {
        "api_access_token": settings.chatwoot_api_access_token,
        "Content-Type": "application/json",
    }


def send_message(conversation_id: int, content: str, private: bool = False) -> None:
    url = (
        f"{settings.chatwoot_base_url}/api/v1/accounts/"
        f"{settings.chatwoot_account_id}/conversations/{conversation_id}/messages"
    )
    payload = {"content": content, "message_type": "outgoing", "private": private}
    with httpx.Client(timeout=15) as client:
        r = client.post(url, headers=_headers(), json=payload)
        r.raise_for_status()


def escalate_to_human(conversation_id: int, reason: str) -> None:
    """Leave a private note for the human agent and flip the conversation
    back to 'open' so it leaves the bot queue."""
    notes_url = (
        f"{settings.chatwoot_base_url}/api/v1/accounts/"
        f"{settings.chatwoot_account_id}/conversations/{conversation_id}/messages"
    )
    status_url = (
        f"{settings.chatwoot_base_url}/api/v1/accounts/"
        f"{settings.chatwoot_account_id}/conversations/{conversation_id}/toggle_status"
    )
    with httpx.Client(timeout=15) as client:
        client.post(
            notes_url,
            headers=_headers(),
            json={"content": f"🤖 Escalated: {reason}", "message_type": "outgoing", "private": True},
        )
        client.post(status_url, headers=_headers(), json={"status": "open"})