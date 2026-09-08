"""
dashboard.py — the screen someone actually watches.

Everything up to now printed to a terminal. That's fine for you, but it's
not what a security desk looks like, and it's not where you can hit cancel
on a countdown.

Run it:
    python dashboard.py                 dashboard only, for testing
    python run_live.py --dashboard      camera feeding into it

Then open http://localhost:5000
"""
import base64
import sqlite3
import threading
import time
from collections import deque

import config as C


# The whole page in one string. No build step, no npm, no node_modules.
PAGE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Security Monitor</title>
<script src="https://cdn.socket.io/4.6.0/socket.io.min.js"></script>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    background: #141821; color: #e8ecf1; font-size: 14px;
  }
  header {
    padding: 14px 22px; background: #1b2029;
    border-bottom: 1px solid #2b3240;
    display: flex; align-items: center; justify-content: space-between;
  }
  h1 { font-size: 16px; font-weight: 600; letter-spacing: .3px; }
  .status { display: flex; gap: 18px; align-items: center; font-size: 12px; color: #8a94a6; }
  .dot { width: 8px; height: 8px; border-radius: 50%; background: #2f6f4a; display: inline-block; margin-right: 6px; }
  .dot.off { background: #7a3b34; }

  .grid { display: grid; grid-template-columns: 1fr 380px; gap: 16px; padding: 16px; }
  @media (max-width: 1000px) { .grid { grid-template-columns: 1fr; } }

  .panel { background: #1b2029; border: 1px solid #2b3240; border-radius: 10px; overflow: hidden; }
  .panel h2 {
    font-size: 12px; font-weight: 600; text-transform: uppercase;
    letter-spacing: .8px; color: #8a94a6;
    padding: 11px 16px; border-bottom: 1px solid #2b3240;
  }

  #feed { width: 100%; display: block; background: #0d1015; min-height: 300px; }
  .nofeed { padding: 60px 20px; text-align: center; color: #5e6878; font-size: 13px; }

  .counts { display: flex; gap: 10px; padding: 14px 16px; }
  .count { flex: 1; text-align: center; padding: 12px 6px; border-radius: 8px; }
  .count .n { font-size: 26px; font-weight: 600; line-height: 1; }
  .count .l { font-size: 10px; text-transform: uppercase; letter-spacing: .8px; margin-top: 5px; opacity: .85; }
  .red { background: rgba(176,58,46,.16); color: #e8776b; }
  .yellow { background: rgba(141,110,31,.16); color: #d9b45a; }
  .green { background: rgba(39,103,73,.16); color: #6bbd8e; }

  #alerts { max-height: 460px; overflow-y: auto; }
  .alert { padding: 12px 16px; border-bottom: 1px solid #232935; border-left: 3px solid transparent; }
  .alert.RED { border-left-color: #b03a2e; background: rgba(176,58,46,.07); }
  .alert.YELLOW { border-left-color: #8d6e1f; background: rgba(141,110,31,.06); }
  .alert.GREEN { border-left-color: #276749; }
  .alert .top { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 4px; }
  .alert .who { font-weight: 600; font-size: 13.5px; }
  .alert .when { font-size: 11px; color: #6f7a8c; }
  .alert .what { font-size: 12.5px; color: #a9b3c2; margin-bottom: 3px; }
  .alert .why { font-size: 11.5px; color: #7d8798; font-style: italic; }
  .badge { display: inline-block; font-size: 10px; font-weight: 600; padding: 2px 7px;
           border-radius: 3px; letter-spacing: .5px; margin-right: 7px; }
  .badge.RED { background: #b03a2e; color: #fff; }
  .badge.YELLOW { background: #8d6e1f; color: #fff; }
  .badge.GREEN { background: #276749; color: #fff; }

  #countdown {
    display: none; position: fixed; inset: 0; z-index: 50;
    background: rgba(10,12,16,.93);
    align-items: center; justify-content: center;
  }
  #countdown.on { display: flex; }
  .cdbox { text-align: center; max-width: 460px; padding: 34px; }
  .cdtitle { font-size: 22px; font-weight: 700; color: #e8776b; margin-bottom: 8px; letter-spacing: .5px; }
  .cdwho { font-size: 15px; color: #c3cbd6; margin-bottom: 4px; }
  .cdwhy { font-size: 13px; color: #8a94a6; margin-bottom: 22px; }
  .cdnum { font-size: 78px; font-weight: 700; color: #e8776b; line-height: 1; margin-bottom: 6px; }
  .cdsub { font-size: 12.5px; color: #8a94a6; margin-bottom: 24px; }
  .cdbtn {
    background: #b03a2e; color: #fff; border: none; padding: 15px 42px;
    font-size: 16px; font-weight: 600; border-radius: 8px; cursor: pointer;
    letter-spacing: .4px;
  }
  .cdbtn:hover { background: #c4453a; }

  .empty { padding: 34px 20px; text-align: center; color: #5e6878; font-size: 13px; }
</style>
</head>
<body>

<header>
  <h1>Security Monitor</h1>
  <div class="status">
    <span><span class="dot off" id="dot"></span><span id="conn">connecting</span></span>
    <span id="watching">0 people tracked</span>
  </div>
</header>

<div class="grid">
  <div>
    <div class="panel">
      <h2>Camera</h2>
      <img id="feed" style="display:none">
      <div class="nofeed" id="nofeed">
        No camera connected.<br><br>
        Start it with: <code>python run_live.py --dashboard</code>
      </div>
    </div>
  </div>

  <div>
    <div class="panel" style="margin-bottom:16px">
      <h2>Today</h2>
      <div class="counts">
        <div class="count red"><div class="n" id="nred">0</div><div class="l">red</div></div>
        <div class="count yellow"><div class="n" id="nyellow">0</div><div class="l">yellow</div></div>
        <div class="count green"><div class="n" id="ngreen">0</div><div class="l">green</div></div>
      </div>
    </div>

    <div class="panel">
      <h2>Alerts</h2>
      <div id="alerts"><div class="empty">Nothing yet</div></div>
    </div>
  </div>
</div>

<div id="countdown">
  <div class="cdbox">
    <div class="cdtitle">CALLING EMERGENCY SERVICES</div>
    <div class="cdwho" id="cdwho"></div>
    <div class="cdwhy" id="cdwhy"></div>
    <div class="cdnum" id="cdnum">10</div>
    <div class="cdsub">seconds until the call goes out</div>
    <button class="cdbtn" onclick="stopIt()">CANCEL &mdash; FALSE ALARM</button>
  </div>
</div>

<script>
const sock = io();
let counts = { RED: 0, YELLOW: 0, GREEN: 0 };
let pending = null, ticker = null;

sock.on("connect", () => {
  document.getElementById("dot").classList.remove("off");
  document.getElementById("conn").textContent = "connected";
});
sock.on("disconnect", () => {
  document.getElementById("dot").classList.add("off");
  document.getElementById("conn").textContent = "disconnected";
});

sock.on("frame", d => {
  const img = document.getElementById("feed");
  img.src = d.image;
  img.style.display = "block";
  document.getElementById("nofeed").style.display = "none";
  document.getElementById("watching").textContent =
    d.people + (d.people === 1 ? " person tracked" : " people tracked");
});

sock.on("alert", a => {
  counts[a.level] = (counts[a.level] || 0) + 1;
  document.getElementById("nred").textContent = counts.RED;
  document.getElementById("nyellow").textContent = counts.YELLOW;
  document.getElementById("ngreen").textContent = counts.GREEN;
  addAlert(a);
});

sock.on("countdown", a => startCountdown(a));
sock.on("cancelled", () => stopCountdown());
sock.on("called", d => {
  stopCountdown();
  addAlert({ level: "RED", person: d.person, action: "call placed",
             confidence: 1, zone: d.service, why: "reference " + d.ref,
             at: Date.now() / 1000 });
});

function addAlert(a) {
  const list = document.getElementById("alerts");
  const blank = list.querySelector(".empty");
  if (blank) blank.remove();

  const t = new Date(a.at * 1000).toLocaleTimeString();
  const el = document.createElement("div");
  el.className = "alert " + a.level;
  el.innerHTML =
    '<div class="top"><span class="who">' +
    '<span class="badge ' + a.level + '">' + a.level + '</span>' +
    'Person ' + a.person + '</span><span class="when">' + t + '</span></div>' +
    '<div class="what">' + a.action + ' &middot; ' +
    Math.round(a.confidence * 100) + '% &middot; ' + a.zone + '</div>' +
    '<div class="why">' + (a.why || "") + '</div>';

  list.insertBefore(el, list.firstChild);
  while (list.children.length > 60) list.removeChild(list.lastChild);
}

function startCountdown(a) {
  pending = a.person;
  document.getElementById("cdwho").textContent =
    "Person " + a.person + " \\u2014 " + a.action + " in " + a.zone;
  document.getElementById("cdwhy").textContent = a.why || "";
  document.getElementById("countdown").classList.add("on");

  let left = a.seconds || 10;
  document.getElementById("cdnum").textContent = left;
  clearInterval(ticker);
  ticker = setInterval(() => {
    left -= 1;
    document.getElementById("cdnum").textContent = Math.max(0, left);
    if (left <= 0) { clearInterval(ticker); stopCountdown(); }
  }, 1000);
}

function stopCountdown() {
  clearInterval(ticker);
  document.getElementById("countdown").classList.remove("on");
  pending = null;
}

function stopIt() {
  if (pending === null) return;
  fetch("/cancel", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ person: pending, note: "cancelled from dashboard" })
  });
  stopCountdown();
}
</script>
</body>
</html>"""


class Dashboard:
    """
    Runs the web server in the background so the camera loop isn't blocked.

    The live system pushes things in through show_alert, show_countdown
    and show_frame. Cancels come back out through whatever you pass as
    on_cancel.
    """

    def __init__(self, on_cancel=None, port=None):
        from flask import Flask, request, jsonify
        from flask_socketio import SocketIO

        self.on_cancel = on_cancel or (lambda person, note: None)
        self.port = port or C.DASHBOARD_PORT
        self.recent = deque(maxlen=200)
        self._last_frame_at = 0.0

        app = Flask(__name__)
        app.config["SECRET_KEY"] = "surveillance"
        sock = SocketIO(app, cors_allowed_origins="*",
                        async_mode="threading", logger=False,
                        engineio_logger=False)

        @app.route("/")
        def home():
            return PAGE

        @app.route("/cancel", methods=["POST"])
        def cancel():
            body = request.get_json(silent=True) or {}
            person = int(body.get("person", -1))
            note = body.get("note", "")
            self.on_cancel(person, note)
            sock.emit("cancelled", {"person": person})
            return jsonify({"ok": True})

        @app.route("/history")
        def history():
            db = C.LOGS / "dispatch_log.db"
            if not db.exists():
                return jsonify([])          # nothing logged yet, that's fine
            try:
                conn = sqlite3.connect(db)
                rows = conn.execute(
                    "SELECT at, person, action, confidence, level, zone, "
                    "service, outcome FROM calls ORDER BY at DESC LIMIT 100"
                ).fetchall()
                conn.close()
                keys = ["at", "person", "action", "confidence",
                        "level", "zone", "service", "outcome"]
                return jsonify([dict(zip(keys, r)) for r in rows])
            except sqlite3.OperationalError:
                return jsonify([])          # table not created yet
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        self.app = app
        self.sock = sock

    # ── pushing things to the browser ─────────────────────────────────
    def show_alert(self, alert):
        data = {
            "person": alert.person, "action": alert.action,
            "confidence": round(alert.confidence, 3), "level": alert.level,
            "camera": alert.camera, "zone": alert.zone,
            "why": alert.why, "at": alert.at,
        }
        self.recent.append(data)
        self.sock.emit("alert", data)

    def show_countdown(self, alert, seconds=None):
        self.sock.emit("countdown", {
            "person": alert.person, "action": alert.action,
            "zone": alert.zone, "why": alert.why,
            "seconds": seconds or C.OVERRIDE_SECONDS,
        })

    def show_called(self, alert, service, ref):
        self.sock.emit("called", {
            "person": alert.person, "service": service, "ref": ref})

    def show_frame(self, frame, people=0):
        """Throttled so we don't flood the browser."""
        now = time.time()
        if now - self._last_frame_at < 1.0 / C.DASHBOARD_FPS:
            return
        self._last_frame_at = now

        import cv2
        small = cv2.resize(frame, (720, 405))
        ok, buf = cv2.imencode(".jpg", small,
                               [cv2.IMWRITE_JPEG_QUALITY, C.JPEG_QUALITY])
        if not ok:
            return
        self.sock.emit("frame", {
            "image": "data:image/jpeg;base64," +
                     base64.b64encode(buf).decode("utf-8"),
            "people": people,
        })

    # ── running it ────────────────────────────────────────────────────
    def _open_browser(self):
        """
        Pop the page open so nobody has to go hunting for the address.

        Wrapped in a try because on a machine with no desktop there is no
        browser to open, and that shouldn't take the whole system down.
        """
        try:
            import webbrowser
            webbrowser.open(f"http://localhost:{self.port}")
        except Exception:
            pass

    def start_background(self):
        def serve():
            self.sock.run(self.app, host="0.0.0.0", port=self.port,
                          debug=False, allow_unsafe_werkzeug=True)

        thread = threading.Thread(target=serve, daemon=True)
        thread.start()

        # Give the server a moment before pointing a browser at it,
        # otherwise the page loads before anything is listening.
        time.sleep(1.5)
        self._open_browser()

        print(f"\n  dashboard: http://localhost:{self.port}")
        print("  (if the page didn't open, paste that address in a browser)")
        return thread

    def run_forever(self):
        print(f"\n  dashboard: http://localhost:{self.port}")
        print("  Ctrl+C to stop\n")

        # Opened from a background thread because sock.run blocks forever
        # once it starts, so anything after it would never happen.
        threading.Timer(1.5, self._open_browser).start()

        self.sock.run(self.app, host="0.0.0.0", port=self.port,
                      debug=False, allow_unsafe_werkzeug=True)


def main():
    """Run the dashboard alone, with fake alerts, to check it works."""
    import random
    from threat import Alert

    print("=" * 46)
    print("  Dashboard test mode")
    print("=" * 46)

    board = Dashboard(on_cancel=lambda p, n: print(f"  cancelled: person {p}"))
    board.start_background()

    print("  open the page, then watch alerts appear here")
    print("  a red countdown will start after about 15 seconds\n")

    actions = ["normal", "stagger", "sos", "collapse"]
    zones = ["Lobby", "Corridor", "Stairwell", "Entrance"]

    try:
        n = 0
        while True:
            time.sleep(3)
            n += 1
            action = random.choice(actions)
            level = ("GREEN" if action == "normal" else
                     "YELLOW" if action in ("stagger", "sos") else "RED")
            alert = Alert(person=random.randint(1, 4), action=action,
                          confidence=random.uniform(0.6, 0.98),
                          level=level, camera=random.randint(0, 2),
                          zone=random.choice(zones),
                          why="generated for testing")
            board.show_alert(alert)
            print(f"  sent {level} {action}")

            if n == 5:
                big = Alert(person=2, action="collapse", confidence=0.95,
                            level="RED", camera=1, zone="Stairwell",
                            why="held 20 frames at 95% confidence")
                board.show_alert(big)
                board.show_countdown(big)
                print("  countdown started, try the cancel button")

    except KeyboardInterrupt:
        print("\n  stopped")


if __name__ == "__main__":
    main()
