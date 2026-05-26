<img alt="Halgorithem logo" src="assets/Halgorithem.png?raw=true" width="420">

# halgo2 Playground Release

**halgo2** is the playground release of Halgorithem: a local Django website plus REST API for testing AI responses against trusted source material.

It lets you:

- paste trusted source text
- drag in text files
- add website/Wikipedia sources
- ask ChatGPT from the browser
- inspect the ChatGPT response before validation
- verify every extracted claim with Halgorithem
- see claim-level evidence, confidence, and support status

ChatGPT generation and Halgorithem validation are intentionally separate. ChatGPT receives only the prompt you type in the ChatGPT instruction box. It does **not** receive the trusted source stack unless you put source text into that prompt yourself. Halgorithem then checks the visible AI response against your provided sources.

## Quick Start

```bash
cd /Users/reyaanshsinha/Tangible/halgo2
make install
make run
```

Open:

```text
http://127.0.0.1:8000/
```

If you already have the shared project virtualenv installed, you can run against it:

```bash
make run VENV=../Venv/venv3.12
```

## OpenAI Key

The playground supports two ways to use ChatGPT:

1. Paste an OpenAI API key into the **OpenAI API key** textbox in the website.
2. Or start the server with an environment variable:

```bash
OPENAI_API_KEY="sk-..." make run
```

The textbox key is sent only to the local Django `/api/chatgpt` endpoint for that request. It is not stored by the app.

## Make Targets

```bash
make help
make install
make install-dev
make run
make run-semantic
make check
make test
make clean
```

`make install` creates `.venv`, installs `requirements.txt`, downloads `en_core_web_sm`, and installs required NLTK data.

`make run` starts the Django playground with the fast lexical embedder:

```bash
HALGORITHEM_EMBEDDER=lexical .venv/bin/python manage.py runserver 127.0.0.1:8000
```

`make run-semantic` enables model downloads for semantic retrieval:

```bash
HALGORITHEM_ALLOW_MODEL_DOWNLOAD=1 .venv/bin/python manage.py runserver 127.0.0.1:8000
```

## Playground Workflow

1. Add trusted sources in the left panel:
   - paste source text
   - drag UTF-8 text files
   - paste website URLs, one per line

2. Generate or paste an AI response:
   - write a ChatGPT instruction
   - add an OpenAI API key
   - click **Ask ChatGPT**
   - read the generated response in the AI response box

3. Validate:
   - click **Run verification**
   - inspect each claim, status, evidence chunk, score, and confidence

The UI displays Halgorithem's unsupported-source verdict as **UNSUPPORTED**. Internally this maps to the legacy `HALLUCINATION` status, meaning “not supported by the supplied source,” not necessarily “false in the real world.”

## REST API

Health:

```bash
curl http://127.0.0.1:8000/api/health
```

Ask ChatGPT:

```bash
curl -X POST http://127.0.0.1:8000/api/chatgpt \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Explain what BASIC is in two sentences.","api_key":"sk-..."}'
```

Verify source text and response:

```bash
curl -X POST http://127.0.0.1:8000/api/verify \
  -F 'source_text=BASIC was created in 1964 by John Kemeny at Dartmouth College.' \
  -F 'response_text=BASIC was created in 1964 by John Kemeny.'
```

Verify with a website source:

```bash
curl -X POST http://127.0.0.1:8000/api/verify \
  -F 'source_urls=https://en.wikipedia.org/wiki/Indian_independence_movement' \
  -F 'response_text=India gained independence from British rule in 1947.'
```

## Python API

```python
from Halgorithem import Halgorithm

algo = Halgorithm(sentences_per_chunk=2, sentence_overlap=1)

results = algo.compare_to_docs(
    truth_docs=[
        {
            "file_id": 1,
            "file_path": "source.txt",
            "text": "BASIC was created in 1964 by John Kemeny at Dartmouth College.",
        }
    ],
    ai_output="BASIC was created in 1972 by NASA.",
)

for result in results:
    print(result["status"], result["claim"], result["reason"])
```

Verify files on disk:

```python
algo.compare_to_files(
    truth_file_paths=["sources/basic.txt"],
    ai_output="BASIC was created by NASA.",
)
```

## CLI Usage

Classic JSON CLI:

```bash
python main.py --document doc.txt --response response.txt
```

Interactive terminal UI:

```bash
python tui.py
```

Benchmark runner:

```bash
python bench.py
```

Stdlib HTTP server, kept for lightweight local use:

```bash
python server.py --host 127.0.0.1 --port 8765
```

## Verification Modes

Fast deterministic mode:

```bash
HALGORITHEM_EMBEDDER=lexical make run
```

Semantic mode with local cached models:

```bash
HALGORITHEM_EMBEDDER=semantic python tui.py
```

Allow model downloads:

```bash
HALGORITHEM_ALLOW_MODEL_DOWNLOAD=1 python tui.py
```

Halgorithem falls back to local deterministic checks when heavyweight models are unavailable and reports that in diagnostics.

## Output Schema

Every claim result includes:

```python
{
    "claim": str,
    "status": "SUPPORTED | WEAK_SUPPORT | CONTRADICTION | HALLUCINATION | UNVERIFIABLE_DENIAL | ERROR",
    "confidence": float,
    "score": float,
    "matched_source": str | None,
    "matched_chunk_id": int | None,
    "matched_chunk": str,
    "chunk_text": str,
    "evidence": list,
    "unsupported_terms": list[str],
    "reason": str,
    "warning": str | None,
}
```

## Verdict Meanings

| Verdict | Meaning |
|:--|:--|
| `SUPPORTED` | Strong evidence is present in the supplied sources. |
| `WEAK_SUPPORT` | Related evidence exists, but the claim is inferential or not fully direct. |
| `CONTRADICTION` | Relevant source evidence conflicts with the claim. |
| `HALLUCINATION` / UI `UNSUPPORTED` | The claim lacks adequate support in the supplied sources. |
| `UNVERIFIABLE_DENIAL` | The claim denies a fact or entity absent from the sources; absence alone cannot prove it. |
| `ERROR` | The verifier could not parse or evaluate the claim. |

## Development

Run checks:

```bash
make check
make test
```

The test suite is intended to run without network access. Some optional ML libraries may emit background network warnings when installed, but the tests use local documents and deterministic fallbacks.

## Project Layout

```text
Halgorithem/           Core verifier package
verifier/              Django playground app
halgo2_site/           Django project settings
static/                Legacy static client
tests/                 Unit tests
Makefile              Install, run, and test workflow
requirements.txt      Runtime and playground dependencies
```
