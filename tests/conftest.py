import os

# Tests never call a real LLM: the template fallback is exercised instead.
os.environ["LLM_PROVIDER"] = "none"
os.environ["LLM_FALLBACKS"] = ""
os.environ["LLM_MIN_INTERVAL_S"] = "0"
