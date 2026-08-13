"""Local dev server with CORS headers for testing the blog integration."""

import http.server
import functools
import sys
from pathlib import Path


class CORSHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8081
    directory = Path(__file__).parent
    handler = functools.partial(CORSHandler, directory=str(directory))
    server = http.server.HTTPServer(("", port), handler)
    print(f"Serving {directory} on http://localhost:{port} (with CORS)")
    print(f"Data URL: http://localhost:{port}/dashboard/data/")
    server.serve_forever()
