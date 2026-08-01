"""Zero-setup terminal demo: talk to the agent directly, no Chatwoot needed.
"""
from app.agent import run_agent


def main():
    print("AI Support Agent — type 'quit' to exit\n")
    print("Try: 'where's my order 1001', 'what's your return policy',")
    print("     'I want a refund for a damaged item', 'I want to speak to a human'\n")

    history: list[dict] = []
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"quit", "exit"}:
            break
        if not user_input:
            continue

        escalated = {"flag": False}

        def _on_escalate(reason: str):
            escalated["flag"] = True
            print(f"  [system: escalation triggered — reason: {reason}]")

        reply, history = run_agent(user_input, history, on_escalate=_on_escalate)
        print(f"Agent: {reply}\n")


if __name__ == "__main__":
    main()
