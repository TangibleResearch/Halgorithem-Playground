import json
import os
from functools import lru_cache
from tempfile import TemporaryDirectory
from urllib.parse import urlparse

from Halgorithem.core import Halgorithm
from Halgorithem.web import scrape_url_texts


MAX_TEXT_CHARS = 120_000
MAX_FILE_CHARS = 80_000
MAX_URLS = 8


@lru_cache(maxsize=1)
def algorithm():
    return Halgorithm()


def parse_json_request(request):
    if not request.body:
        return {}
    try:
        return json.loads(request.body.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("Request body must be valid JSON.") from exc


def parse_urls(raw):
    if isinstance(raw, list):
        candidates = raw
    else:
        candidates = str(raw or "").replace(",", "\n").splitlines()
    urls = []
    for candidate in candidates:
        value = str(candidate).strip()
        if not value:
            continue
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(f"Invalid URL: {value}")
        urls.append(value)
    return urls[:MAX_URLS]


def read_uploaded_files(files):
    records = []
    for index, uploaded in enumerate(files, 1):
        content_type = uploaded.content_type or ""
        name = uploaded.name or f"upload_{index}.txt"
        if content_type and not (
            content_type.startswith("text/")
            or content_type in {"application/json", "application/xml", "application/csv"}
        ):
            raise ValueError(f"{name} is not a supported text file.")
        raw = uploaded.read()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"{name} must be UTF-8 text.") from exc
        text = text[:MAX_FILE_CHARS]
        if text.strip():
            records.append({"file_id": index, "file_path": name, "text": text})
    return records


def collect_source_docs(*, source_text="", source_name="pasted_source", urls=None, files=None):
    docs = []
    source_text = str(source_text or "").strip()
    if source_text:
        docs.append({"file_id": len(docs) + 1, "file_path": source_name or "pasted_source", "text": source_text[:MAX_TEXT_CHARS]})

    if files:
        for record in read_uploaded_files(files):
            record["file_id"] = len(docs) + 1
            docs.append(record)

    scraped = []
    urls = parse_urls(urls)
    if urls:
        with TemporaryDirectory(prefix="halgo2-web-") as tmp:
            scraped = scrape_url_texts(urls, output_dir=tmp)
        for record in scraped:
            docs.append({
                "file_id": len(docs) + 1,
                "file_path": record["url"],
                "text": record["text"][:MAX_FILE_CHARS],
            })

    if not docs:
        raise ValueError("Add source text, drop a text file, or provide at least one URL.")
    return docs, scraped


def summarize_results(results):
    counts = {}
    for result in results:
        status = result.get("status", "UNKNOWN")
        counts[status] = counts.get(status, 0) + 1
    total = len(results)
    supported = counts.get("SUPPORTED", 0)
    weak = counts.get("WEAK_SUPPORT", 0)
    issues = counts.get("HALLUCINATION", 0) + counts.get("CONTRADICTION", 0) + counts.get("ERROR", 0)
    confidence = ((supported + 0.5 * weak) / total) if total else 0.0
    return {
        "total_claims": total,
        "supported": supported,
        "weak_support": weak,
        "issues": issues,
        "status_counts": counts,
        "confidence": round(confidence, 4),
    }


def generate_chatgpt_response(instruction="", model=None, api_key=None):
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Add an OpenAI API key in the ChatGPT section or start the server with OPENAI_API_KEY.")

    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    model = model or os.getenv("HALGO2_OPENAI_MODEL", "gpt-4o-mini")
    prompt = instruction.strip()
    if not prompt:
        raise ValueError("Add a ChatGPT instruction before generating a response.")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a careful assistant. Answer the user's request directly. "
                    "Do not assume access to private source material unless the user provides it in the prompt."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content or ""


def verify_payload(*, source_text="", response_text="", source_name="pasted_source", urls=None, files=None, threshold=0.30):
    docs, scraped = collect_source_docs(source_text=source_text, source_name=source_name, urls=urls, files=files)

    if not str(response_text or "").strip():
        raise ValueError("Add an AI response or generate one with ChatGPT first.")

    threshold = float(threshold)
    results = algorithm().compare_to_docs(docs, response_text, threshold=threshold)
    return {
        "summary": summarize_results(results),
        "results": results,
        "diagnostics": algorithm().diagnostics,
        "sources": [{"name": doc["file_path"], "characters": len(doc["text"])} for doc in docs],
        "scraped": [{"url": item["url"], "characters": len(item.get("text", ""))} for item in scraped],
    }
