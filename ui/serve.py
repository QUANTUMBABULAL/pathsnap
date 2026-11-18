"""Serve the evaluator UI (Laptop 2).

    python ui/serve.py            # http://0.0.0.0:5500
    python ui/serve.py --port 8080

Then open the printed URL in a browser and enter Laptop 1's API address
(e.g. http://192.168.1.20:8000) in the "API connection" box.
"""
import argparse
import functools
import http.server
import os
import socket

here = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=5500)
args = ap.parse_args()

handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=here)
try:
    lan = socket.gethostbyname(socket.gethostname())
except OSError:
    lan = "127.0.0.1"
print(f"PathSnap UI on http://localhost:{args.port}  (LAN: http://{lan}:{args.port})")
http.server.ThreadingHTTPServer(("0.0.0.0", args.port), handler).serve_forever()
