import argparse
import json
import mimetypes
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from .core import Halgorithm


ROOT = Path(__file__).resolve().parents[1]
STATIC_ROOT = ROOT / "static"
CLIENT_FILE = STATIC_ROOT / "index.html"


class VerificationService:
    def __init__(self):
        self.algorithm = Halgorithm()

    def verify(self, payload):
        source_text = payload.get("source_text") or payload.get("document_text") or ""
        response_text = payload.get("response_text") or payload.get("ai_output") or ""
        source_name = payload.get("source_name") or "client_input"
        threshold = float(payload.get("threshold", 0.30))

        if not source_text.strip():
            raise ValueError("source_text is required.")
        if not response_text.strip():
            raise ValueError("response_text is required.")

        docs = [{"file_id": 1, "file_path": source_name, "text": source_text}]
        results = self.algorithm.compare_to_docs(docs, response_text, threshold=threshold)
        summary = summarize_results(results)
        return {
            "summary": summary,
            "results": results,
            "diagnostics": self.algorithm.diagnostics,
        }


def summarize_results(results):
    counts = {}
    for result in results:
        status = result.get("status", "UNKNOWN")
        counts[status] = counts.get(status, 0) + 1

    total = len(results)
    supported = counts.get("SUPPORTED", 0)
    weak = counts.get("WEAK_SUPPORT", 0)
    risk = (
        counts.get("HALLUCINATION", 0)
        + counts.get("CONTRADICTION", 0)
        + counts.get("ERROR", 0)
    )
    confidence = ((supported + 0.5 * weak) / total) if total else 0.0

    return {
        "total_claims": total,
        "supported": supported,
        "weak_support": weak,
        "issues": risk,
        "status_counts": counts,
        "confidence": round(confidence, 4),
    }


def make_handler(service):
    class HalgorithemRequestHandler(BaseHTTPRequestHandler):
        server_version = "halgo2/1.0"

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/health":
                self.send_json({"ok": True, "service": "halgo2"})
                return
            if path in {"/", "/index.html"}:
                self.send_file(CLIENT_FILE)
                return
            if path.startswith("/static/"):
                self.send_file(STATIC_ROOT / path.removeprefix("/static/"))
                return
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")

        def do_HEAD(self):
            path = urlparse(self.path).path
            if path in {"/", "/index.html"}:
                self.send_file(CLIENT_FILE, include_body=False)
                return
            if path.startswith("/static/"):
                self.send_file(STATIC_ROOT / path.removeprefix("/static/"), include_body=False)
                return
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")

        def do_POST(self):
            path = urlparse(self.path).path
            if path != "/api/verify":
                self.send_error(HTTPStatus.NOT_FOUND, "Not found")
                return

            try:
                payload = self.read_json()
                response = service.verify(payload)
            except ValueError as exc:
                self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            except Exception as exc:
                self.send_json({"error": "Verification failed.", "detail": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)
            else:
                self.send_json(response)

        def do_OPTIONS(self):
            self.send_response(HTTPStatus.NO_CONTENT)
            self.send_cors_headers()
            self.end_headers()

        def read_json(self):
            length = int(self.headers.get("Content-Length", "0") or "0")
            if length <= 0:
                raise ValueError("Request body must be JSON.")
            raw = self.rfile.read(length)
            try:
                return json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError as exc:
                raise ValueError("Request body must be valid JSON.") from exc

        def send_json(self, data, status=HTTPStatus.OK):
            body = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
            self.send_response(status)
            self.send_cors_headers()
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def send_file(self, path, include_body=True):
            if not path.exists() or not path.is_file():
                self.send_error(HTTPStatus.NOT_FOUND, "Not found")
                return
            body = path.read_bytes()
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            self.send_response(HTTPStatus.OK)
            self.send_cors_headers()
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if include_body:
                self.wfile.write(body)

        def send_cors_headers(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")

        def log_message(self, fmt, *args):
            print(f"{self.address_string()} - {fmt % args}")

    return HalgorithemRequestHandler


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the halgo2 Halgorithem HTTP server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)

    service = VerificationService()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(service))
    print(f"halgo2 server running at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping halgo2 server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
