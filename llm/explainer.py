"""LLM explanation of engine output (spec 11): the LLM sees only the facts JSON, a number
verifier checks the reply, one regeneration is allowed, then the template is used."""
from __future__ import annotations

import json

from llm.facts import build_facts
from llm.providers import LLMChain, ProviderError
from llm.templates import render
from llm.verifier import verify

SYSTEM = """You explain a home energy assessment to a homeowner in plain English.
Rules (strict):
- Use ONLY facts from the FACTS JSON. Copy every number exactly as it appears there. Never calculate, add, convert or estimate new numbers.
- If something is not in FACTS, say it cannot be estimated. Do not guess.
- If a requested retrofit is listed as not suitable, explain the reason and do NOT give savings for it.
- If electricity use rises (a heat pump), say so plainly and explain why, alongside the gas and carbon saving.
- If FACTS asks for missing information, ask those questions.
- All money is in Indian rupees (₹). Write amounts with the digits exactly as in FACTS; never use lakh, crore, k or pounds.
- Savings are estimates, not guarantees; say so once.
- 90 to 180 words. No headings. No tables."""


def _prompt(facts: dict, question: str) -> str:
    q = question.strip() or "Explain my results."
    return f"USER QUESTION: {q}\n\nFACTS:\n{json.dumps(facts, indent=1, ensure_ascii=False)}"


def explain(results: dict, question: str = "", requested: list[str] | None = None, diff: dict | None = None,
            question_type: str = "general", chain: LLMChain | None = None) -> dict:
    facts = build_facts(results, question, requested, diff)
    chain = chain or LLMChain()
    attempts, errors = [], []
    prompt = _prompt(facts, question)
    for attempt in range(2):
        try:
            text, provider = chain.generate(SYSTEM, prompt)
        except ProviderError as exc:
            errors.append(str(exc))
            break
        check = verify(text, facts, question)
        attempts.append({"provider": provider, "ok": check["ok"], "unsupported": check["unsupported"]})
        if check["ok"]:
            return {"text": text.strip(), "source": provider, "verified": True, "attempts": attempts, "errors": errors, "facts": facts}
        prompt = (_prompt(facts, question) + "\n\nYour previous answer used numbers that are not in FACTS: "
                  + ", ".join(check["unsupported"]) + ". Rewrite it using only numbers copied from FACTS.")
    text = render(facts, question_type)
    check = verify(text, facts, question)
    return {"text": text, "source": "template", "verified": check["ok"], "attempts": attempts, "errors": errors,
            "unsupported": check["unsupported"], "facts": facts}
