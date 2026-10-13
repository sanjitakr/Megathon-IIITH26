# LLM implementation

**Role.** The LLM never calculates. It (1) turns a customer's message into a validated scenario and (2) explains numbers the engine has already computed. A what-if always reruns the planner, savings, carbon and value first (`engine/whatif.py`); the LLM then explains the new results.

## Models and providers (free tiers only)

`llm/providers.py` is one adapter. `LLM_PROVIDER` (default `gemini`) picks the primary, `LLM_FALLBACKS` lists the next ones, and deterministic templates are always the last resort, so the demo never depends on an API being up.

| Provider | Default model | Notes |
|---|---|---|
| Google Gemini API (free tier) | `gemini-flash-latest`, fallback `gemini-flash-lite-latest` | `-latest` aliases survive model retirement; on 404 the next model is used; on 429/503 the lite model is tried, then the call waits the suggested delay and retries. `GEMINI_THINKING_LEVEL=low` keeps answers ≈3 s. |
| Groq / OpenRouter / Hugging Face | Llama 3.1 8B / Qwen 2.5 7B | OpenAI-compatible chat API |
| Ollama (local) | `qwen2.5:7b` | JSON-constrained output |
| Templates | – | `llm/templates.py`, every number copied from the facts |

Calls are spaced (`LLM_MIN_INTERVAL_S`, default 4 s) to stay within free per-minute limits. Cost: free.

## Pipeline

1. **Parse** (`llm/parser.py`): system prompt + `SCENARIO_JSON_SPEC` (`llm/schema.py`), JSON mode, validated by the Pydantic `Scenario` (retrofits to add/remove, solar kWp, budget in ₹, goal, prices in ₹/kWh, grid-carbon year, question type). Invalid output or no provider → rule-based parser (understands "₹3,50,000", "2 lakh", "₹11 per kWh").
2. **Engine rerun** for the scenario.
3. **Facts** (`llm/facts.py`): a compact, pre-rounded subset of the results JSON — score, key issues, baseline per fuel, recommended plan with per-fuel savings and ranges, unsuitable retrofits with reasons, warnings, value range or the reason it cannot be estimated, change versus the previous plan. Money in ₹. No name or address is sent.
4. **Explain** (`llm/explainer.py`) with this system prompt:

```
You explain a home energy assessment to a homeowner in plain English.
Rules (strict):
- Use ONLY facts from the FACTS JSON. Copy every number exactly as it appears there. Never calculate, add, convert or estimate new numbers.
- If something is not in FACTS, say it cannot be estimated. Do not guess.
- If a requested retrofit is listed as not suitable, explain the reason and do NOT give savings for it.
- If electricity use rises (a heat pump), say so plainly and explain why, alongside the gas and carbon saving.
- If FACTS asks for missing information, ask those questions.
- All money is in Indian rupees (₹). Write amounts with the digits exactly as in FACTS; never use lakh, crore, k or pounds.
- Savings are estimates, not guarantees; say so once.
- 90 to 180 words. No headings. No tables.
```

5. **Verify** (`llm/verifier.py`): every number in the reply must match a number in the facts or in the user's question, allowing rounding at the written precision, sign, percentage and currency formatting (₹, Rs., Indian digit grouping). Units such as CO₂, m², P10 are ignored. On failure the LLM is asked once to rewrite using only facts numbers (the unsupported numbers are listed); a second failure falls back to the template.

## Evaluation approach

- **Automatic** (`make llm-eval`): 57 held-out questions over 7 home profiles (`llm/eval_questions.json`) — explanations, what-ifs, budgets, goals, price changes, heat-pump electricity rise, infeasible retrofits (listed building, tenant, cavity vs solid wall), missing information, conflicting data, band-C targets, solar sizes, value. Reports verifier pass rate, template-fallback rate and unsupported-number rate; writes `docs/llm_audit.csv`.
- **Manual audit**: each LLM-written answer is checked against the exact facts it was given and marked grounded (Y/N), factually accurate (Y/N) and number of unsupported claims (`make llm-audit`). Targets: groundedness ≥ 90%, accuracy ≥ 90%, unsupported claims < 5%.
- **Result (Gemini, 57 answers):** verifier pass rate 100%, template fallback 0%; manual audit groundedness 70.2%, factual accuracy 78.9%, unsupported claims 5.1% of sentences — below target. Explanations, infeasible requests, missing data and conflicts were handled correctly; failures were unrequested follow-up questions, claiming a re-run what-if "cannot be estimated", sign wording ("rises by -1,013 kWh") and confusing score with EPC band. The audit was AI-assisted and reviewed by the team; per-answer reasons are in the `AUDIT_note` column.
- **Limitation:** the verifier checks numbers, not wording; spelled-out numbers are not checked — which is why the manual audit exists.
