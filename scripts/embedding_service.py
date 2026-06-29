import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from embeddings import DATASET_PATHS, run_embedding_lookup


class EmbeddingHandler(BaseHTTPRequestHandler):
    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json(200, {"status": "ok"})
            return
        self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path != "/embed":
            self._send_json(404, {"error": "not found"})
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(content_length) or b"{}")
            doc_id = payload["doc_id"]
            dataset = payload.get("dataset", "pan2011")
            build_index = bool(payload.get("build_index", False))

            if dataset not in DATASET_PATHS:
                self._send_json(400, {"error": f"Unknown dataset: {dataset}"})
                return

            result = run_embedding_lookup(
                doc_id=doc_id,
                dataset=dataset,
                build_index=build_index,
            )
            self._send_json(200, result)
        except KeyError:
            self._send_json(400, {"error": "Missing required field: doc_id"})
        except Exception as exc:
            self._send_json(500, {"error": str(exc)})


def main() -> None:
    server = ThreadingHTTPServer(("0.0.0.0", 8000), EmbeddingHandler)
    print("Embedding service listening on :8000", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
