"""
dispatch.py — placing the emergency call.

Nothing here fires without a countdown first. Ten seconds isn't long,
but it's enough for whoever is watching to hit cancel, and it means
no call ever goes out that a human couldn't have stopped.

SAFETY: while TEST_MODE is True in config.py, every call goes to your
own phone. Leave it that way until your final demo.
"""
import sqlite3
import threading
import time

import config as C
from threat import Alert


# ── Audit trail ───────────────────────────────────────────────────────────
DB = C.LOGS / "dispatch_log.db"


def _setup_db():
    conn = sqlite3.connect(DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS calls (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            at         REAL,
            person     INTEGER,
            action     TEXT,
            confidence REAL,
            level      TEXT,
            camera     INTEGER,
            zone       TEXT,
            service    TEXT,
            outcome    TEXT,
            call_ref   TEXT,
            note       TEXT
        )
    """)
    conn.commit()
    conn.close()


def record(alert, service, outcome, call_ref="", note=""):
    """Write down what happened. Every decision, including the ones cancelled."""
    conn = sqlite3.connect(DB)
    conn.execute(
        "INSERT INTO calls (at, person, action, confidence, level, camera, "
        "zone, service, outcome, call_ref, note) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (alert.at, alert.person, alert.action, alert.confidence, alert.level,
         alert.camera, alert.zone, service, outcome, call_ref, note))
    conn.commit()
    conn.close()


# ── The call itself ───────────────────────────────────────────────────────
def spoken_message(alert, service):
    return (f"Automated safety alert. "
            f"A possible {alert.action} has been detected "
            f"on camera {alert.camera}, {alert.zone}. "
            f"Confidence {int(alert.confidence * 100)} percent. "
            f"This is an automated system. Please respond.")


def place_call(alert, service):
    """
    Dial through Twilio and read out the alert.
    Returns a reference string for the log.
    """
    number = C.YOUR_PHONE if C.TEST_MODE else C.REAL_NUMBERS.get(
        service, C.YOUR_PHONE)
    message = spoken_message(alert, service)

    print()
    print(f"  calling {service} on {number}")
    print(f"  saying: {message}")
    if C.TEST_MODE:
        print("  (TEST_MODE is on, so this goes to your own phone)")
    print()

    try:
        from twilio.rest import Client
        from twilio.twiml.voice_response import VoiceResponse

        speech = VoiceResponse()
        speech.say(message, voice="alice", language="en-GB")
        speech.pause(length=1)
        speech.say(message, voice="alice", language="en-GB")

        client = Client(C.TWILIO_SID, C.TWILIO_TOKEN)
        call = client.calls.create(twiml=str(speech),
                                   to=number, from_=C.TWILIO_FROM)
        print(f"  connected, reference {call.sid}")
        return call.sid

    except ImportError:
        print("  twilio isn't installed, so nothing was actually dialled")
        print("  install it with:  pip install twilio")
        return "not-installed"
    except Exception as e:
        print(f"  the call failed: {e}")
        return f"failed: {e}"


# ── Deciding whether to call ──────────────────────────────────────────────
class Dispatcher:
    """
    Handles the countdown, the cancel, and the call.

    RED    -> countdown starts, call goes out unless cancelled
    YELLOW -> tell the operator, wait for them to decide
    GREEN  -> write it down and say nothing
    """

    def __init__(self, on_alert=None, on_call=None, on_cancel=None):
        self.on_alert  = on_alert  or (lambda *a: None)
        self.on_call   = on_call   or (lambda *a: None)
        self.on_cancel = on_cancel or (lambda *a: None)
        self._timers = {}
        _setup_db()
        print(f"  dispatcher ready (TEST_MODE = {C.TEST_MODE})")

    def handle(self, alert):
        if alert.level == "RED":
            self._start_countdown(alert)
        elif alert.level == "YELLOW":
            record(alert, C.SERVICE_FOR.get(alert.action) or "-", "operator_alerted")
            self.on_alert(alert)
        else:
            record(alert, "-", "logged")

    def _start_countdown(self, alert):
        person = alert.person

        # Already counting down for this person, leave it alone
        existing = self._timers.get(person)
        if existing and existing.is_alive():
            return

        service = C.SERVICE_FOR.get(alert.action) or "AMBULANCE"

        print()
        print("  " + "!" * 42)
        print(f"  RED  person {person}  {alert.action}  "
              f"{alert.confidence:.0%}  {alert.zone}")
        print(f"  {alert.why}")
        print(f"  calling {service} in {C.OVERRIDE_SECONDS}s "
              f"unless someone cancels")
        print("  " + "!" * 42)

        record(alert, service, "countdown_started")
        self.on_alert(alert)

        timer = threading.Timer(C.OVERRIDE_SECONDS, self._go, (alert, service))
        timer.daemon = True
        self._timers[person] = timer
        timer.start()

    def _go(self, alert, service):
        ref = place_call(alert, service)
        record(alert, service, "called", call_ref=ref)
        self.on_call(alert, service, ref)
        self._timers.pop(alert.person, None)

    def cancel(self, person, note=""):
        """Someone hit cancel. Stop the countdown."""
        timer = self._timers.pop(person, None)
        if timer and timer.is_alive():
            timer.cancel()
            print(f"  cancelled for person {person}. {note}")
            record(Alert(person=person, action="", confidence=0.0,
                         level="RED", why="cancelled by operator"),
                   "-", "cancelled", note=note)
            self.on_cancel(person, note)
            return True
        return False

    def call_now(self, alert, service):
        """Operator decided to escalate a YELLOW themselves."""
        ref = place_call(alert, service)
        record(alert, service, "called_manually", call_ref=ref)
        self.on_call(alert, service, ref)
