"""Evaluation harness for the RAG + agent pipeline.
Two things are measured:
1. Retrieval quality: for each labeled question, does the retrieved
   context actually contain the source doc we know has the answer?
2. Answer quality : an LLM-as-judge scores the agent's final reply against
   a reference answer on a 1-5 rubric (correctness, groundedness, tone).
"""
from __future__ import annotations

import json
import time
import statistics
from dotenv import load_dotenv
load_dotenv()

from app.agent import run_agent
from app.llm_backends import call_llm
from app.rag import retrieve

# Labeled test set: question, the doc that SHOULD be retrieved, and a
# reference answer to judge against.
TEST_SET = [
    {
        "question": "How long do I have to return an item?",
        "expected_source": "returns_policy.md",
        "reference_answer": "Items can be returned within 30 days of delivery for a full refund if unused and in original packaging.",
    },
    {
        "question": "Do you ship internationally?",
        "expected_source": "shipping_policy.md",
        "reference_answer": "Yes, to Canada, UK, and EU, but the customer pays customs/import duties.",
    },
    {
        "question": "Can I combine two promo codes on one order?",
        "expected_source": "account_payment_faq.md",
        "reference_answer": "No, only one promo code can be applied per order.",
    },
    {
        "question": "My package says delivered but I never got it, what do I do?",
        "expected_source": "shipping_policy.md",
        "reference_answer": "Wait 48 hours since it may be scanned early, then contact support with the order number to open a carrier investigation.",
    },
    {
        "question": "I received a broken item, can I get a replacement without shipping it back?",
        "expected_source": "returns_policy.md",
        "reference_answer": "Contact support with photos within 7 days of delivery for an expedited replacement or refund with no return shipping required.",
    },
]

JUDGE_PROMPT = """You are grading a customer support agent's reply.

Question: {question}
Reference answer (ground truth): {reference}
Agent's actual reply: {reply}

Score the agent's reply from 1-5 on:
- correctness: does it match the reference answer's facts? (no contradictions, no fabrication)
- groundedness: does it avoid making up policy details not in the reference?

Respond with ONLY a JSON object: {{"correctness": <1-5>, "groundedness": <1-5>, "reasoning": "<one sentence>"}}
"""


def eval_retrieval() -> float:
    hits = 0
    for row in TEST_SET:
        results = retrieve(row["question"], k=3)
        sources = {r["source"] for r in results}
        if row["expected_source"] in sources:
            hits += 1
    recall_at_3 = hits / len(TEST_SET)
    return recall_at_3


def judge_reply(question: str, reference: str, reply: str) -> dict:
    prompt = JUDGE_PROMPT.format(question=question, reference=reference, reply=reply)
    response = call_llm(
        system="You are a strict, concise grading assistant. Respond with only the requested JSON.",
        messages=[{"role": "user", "content": prompt}],
        tools=[],
    )
    text = "".join(b["text"] for b in response["content"] if b["type"] == "text")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"correctness": None, "groundedness": None, "reasoning": "judge output not parseable"}


def eval_end_to_end() -> list[dict]:
    results = []
    for row in TEST_SET:
        reply, _ = run_agent(row["question"], [])
        score = judge_reply(row["question"], row["reference_answer"], reply)
        results.append({"question": row["question"], "reply": reply, **score})
        time.sleep(2)
    return results


def main():
    print("=== Retrieval evaluation ===")
    recall = eval_retrieval()
    print(f"Recall@3: {recall:.2%} ({int(recall * len(TEST_SET))}/{len(TEST_SET)} correct doc retrieved)\n")

    print("=== End-to-end answer quality (LLM-as-judge) ===")
    results = eval_end_to_end()
    for r in results:
        print(f"- Q: {r['question']}")
        print(f"  Correctness: {r['correctness']}/5  Groundedness: {r['groundedness']}/5")
        print(f"  Reasoning: {r['reasoning']}\n")

    correctness_scores = [r["correctness"] for r in results if r["correctness"] is not None]
    groundedness_scores = [r["groundedness"] for r in results if r["groundedness"] is not None]
    if correctness_scores:
        print(f"Avg correctness: {statistics.mean(correctness_scores):.2f}/5")
    if groundedness_scores:
        print(f"Avg groundedness: {statistics.mean(groundedness_scores):.2f}/5")


if __name__ == "__main__":
    main()
