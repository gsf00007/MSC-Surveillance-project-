"""
setup_calls.py — get the emergency calls working, safely.

Run:   python setup_calls.py

Checks your Twilio details, then offers to ring your own phone with the
same message the system would use for a real alert.

It will not let you dial a real emergency number from here. That only
happens from the live system, with TEST_MODE off, which you should not
touch until your final demo.
"""
import sys

import config as C


LINE = "=" * 52


def explain_setup():
    print("""
  You need a Twilio account. The free trial covers a few hundred
  test calls, which is far more than you'll need.

    1. Sign up at twilio.com/try-twilio
    2. Verify your own mobile number when it asks
    3. From the Console dashboard, copy:
         Account SID    starts with AC
         Auth Token     click to reveal
    4. Get a phone number:
         Phone Numbers -> Manage -> Buy a number
         Trial credit covers this

  Then fill these in at the bottom of config.py:

    TWILIO_SID   = "AC..."
    TWILIO_TOKEN = "your token"
    TWILIO_FROM  = "+1..."          the Twilio number
    YOUR_PHONE   = "+971..."        your own mobile

  On a trial account you can only call numbers you've verified,
  which is a useful safety net while you're building.
""")


def check_library():
    try:
        import twilio
        print(f"  twilio library    installed ({twilio.__version__})")
        return True
    except ImportError:
        print("  twilio library    MISSING")
        print("                    pip install twilio")
        return False


def check_settings():
    problems = []

    sid = C.TWILIO_SID
    if not sid or sid.startswith("PASTE"):
        print("  account sid       not filled in")
        problems.append("TWILIO_SID")
    elif not sid.startswith("AC"):
        print(f"  account sid       looks wrong (should start AC)")
        problems.append("TWILIO_SID")
    else:
        print(f"  account sid       {sid[:8]}...{sid[-4:]}")

    token = C.TWILIO_TOKEN
    if not token or token.startswith("PASTE"):
        print("  auth token        not filled in")
        problems.append("TWILIO_TOKEN")
    else:
        print(f"  auth token        {'*' * 12}{token[-4:]}")

    for name, value in (("from number", C.TWILIO_FROM),
                        ("your phone", C.YOUR_PHONE)):
        if not value or "X" in value:
            print(f"  {name:<17} not filled in")
            problems.append(name)
        elif not value.startswith("+"):
            print(f"  {name:<17} needs a country code, e.g. +971...")
            problems.append(name)
        else:
            print(f"  {name:<17} {value}")

    return problems


def check_account():
    """Ask Twilio whether the credentials actually work."""
    try:
        from twilio.rest import Client
        client = Client(C.TWILIO_SID, C.TWILIO_TOKEN)
        account = client.api.accounts(C.TWILIO_SID).fetch()
        print(f"  account status    {account.status}")

        numbers = client.incoming_phone_numbers.list(limit=5)
        if numbers:
            print(f"  your numbers      "
                  f"{', '.join(n.phone_number for n in numbers)}")
            owned = [n.phone_number for n in numbers]
            if C.TWILIO_FROM not in owned:
                print(f"                    warning: TWILIO_FROM isn't one "
                      f"of these")
        else:
            print("  your numbers      none yet, buy one in the console")
        return True

    except Exception as e:
        print(f"  connection        failed: {e}")
        return False


def make_test_call():
    """Ring your own phone with a realistic alert message."""
    from twilio.rest import Client
    from twilio.twiml.voice_response import VoiceResponse

    message = ("Automated safety alert. "
               "A possible collapse has been detected "
               "on camera two, Lobby. "
               "Confidence ninety four percent. "
               "This is a test of an automated system. "
               "No action is needed.")

    print(f"\n  calling {C.YOUR_PHONE}")
    print(f"  it will say:\n    {message}\n")

    speech = VoiceResponse()
    speech.say(message, voice="alice", language="en-GB")
    speech.pause(length=1)
    speech.say(message, voice="alice", language="en-GB")

    client = Client(C.TWILIO_SID, C.TWILIO_TOKEN)
    call = client.calls.create(twiml=str(speech),
                               to=C.YOUR_PHONE, from_=C.TWILIO_FROM)

    print(f"  placed, reference {call.sid}")
    print("  your phone should ring shortly")


def main():
    print(LINE)
    print("  Emergency call setup")
    print(LINE)
    print()

    if not check_library():
        print("\n  install it first, then run this again")
        sys.exit(1)

    problems = check_settings()
    if problems:
        print(f"\n  still to fill in: {', '.join(problems)}")
        explain_setup()
        sys.exit(1)

    print()
    if not check_account():
        print("\n  check your SID and token in config.py")
        sys.exit(1)

    print()
    print(LINE)
    if C.TEST_MODE:
        print("  TEST_MODE is ON, so calls go to your own phone.")
        print("  Leave it that way until your final demo.")
    else:
        print("  TEST_MODE IS OFF.")
        print("  The live system would dial REAL emergency numbers.")
        print("  Set TEST_MODE = True in config.py unless you are")
        print("  running a supervised demo right now.")
    print(LINE)

    print()
    answer = input("  Ring your phone with a test alert? (y/n) ").strip().lower()
    if answer == "y":
        try:
            make_test_call()
        except Exception as e:
            print(f"\n  it failed: {e}")
            print("\n  common causes:")
            print("    trial accounts can only call verified numbers")
            print("    the from number must be one you own in Twilio")
            print("    numbers need a country code, like +971501234567")
    else:
        print("  skipped")

    print("\n  done")


if __name__ == "__main__":
    main()
