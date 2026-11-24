"""
Start Neurova and open it in the browser.

    python run.py            (start.bat and start.sh call this for you)

Set PORT to use a different port, for example on a hosting service.
"""
import os
import threading
import time
import urllib.request
import webbrowser

import uvicorn

PORT = int(os.environ.get("PORT", 8000))
URL = f"http://localhost:{PORT}"


def open_browser_when_ready():
    for _ in range(120):
        try:
            urllib.request.urlopen(URL + "/api/health", timeout=1)
            print(f"\n  Neurova is running at {URL}\n  Press Ctrl+C in this window to stop it.\n", flush=True)
            webbrowser.open(URL)
            return
        except OSError:
            time.sleep(0.5)


if __name__ == "__main__":
    hosted = "PORT" in os.environ                      # on a server there is no browser to open
    if not hosted:
        threading.Thread(target=open_browser_when_ready, daemon=True).start()
    uvicorn.run("app.server:app", host="0.0.0.0" if hosted else "127.0.0.1", port=PORT, log_level="warning")
