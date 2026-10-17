# LLOYDS - Green Home Score and Retrofit Recommendation Engine

## Team

* [**Sanjita Kumar**](https://github.com/sanjitakr)
* [**Tanisha P**](https://github.com/TanishaP6174) 
* [**Nandini S**](https://github.com/Nandini-0603) 
* [**B Ananya Pai**](https://github.com/ananyapai585-blip)


## Requirements

- Python 3.11 or 3.12, `make`
- Optional: [uv](https://docs.astral.sh/uv/) (faster install; plain `venv` + `pip` is used if it is missing)
- Optional: a free Google Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey). Without a key the app still runs and uses built-in template answers.

## Setup

Clone the project and cd to the project folder.
```bash
make setup                 # creates .venv and installs dependencies
cp .env.example .env       # optional: paste GEMINI_API_KEY into .env
```

## Build data and models

No datasets or model weights need downloading: everything is generated locally from the code with fixed seeds.

```bash
make all       # dataset, models, held-out predictions, LLM evaluation, evaluation report
```

`make all` takes about 3 minutes, plus about 10 minutes for the LLM evaluation when a Gemini key is set (57 rate-limited calls). Individual steps:

| Command | What it does |
|---|---|
| `make data` | calibrate and generate the synthetic dataset (`data/processed/`) |
| `make train` | train the models (`data/models/`) |
| `make value-demo` | train the property-value model |
| `make predict` | held-out test predictions (`data/predictions/`) |
| `make llm-eval` | run the LLM evaluation questions (overwrites `docs/llm_audit.csv`) |
| `make llm-audit` | score the manual audit columns of `docs/llm_audit.csv` |
| `make evaluate` | write `docs/evaluation_report.md` |

## Run

```bash
make app     # dashboard → http://localhost:8501
make api     # REST API  → http://localhost:8000/docs
make test    # test suite
```

Theme: the dashboard follows your device's light/dark setting; change it with ⋮ (top right) → Settings → Theme.

## Docker

```bash
docker build -t green-home-engine .
docker run -p 8501:8501 -p 8000:8000 --env-file .env green-home-engine
```

## Documentation

- `deliverables/final_report.pdf`: solution, approach, features and results
- `docs/`: scoring method, LLM implementation, architecture, evaluation report, Responsible AI note, data schema and sources

## License

MIT, see [LICENSE](LICENSE).

## AI Disclosure-
The code and system was written and designed with the help of AI.