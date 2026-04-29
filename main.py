#!/usr/bin/env python
"""
Meeting Minutes Pipeline - OpenAI Version
Uses: OpenAI GPT-4o for LLM tasks + OpenAI Whisper for audio transcription
Gmail Draft is created via Google API

Setup:
  1. Install dependencies:
       pip install openai pydub python-dotenv google-auth google-auth-oauthlib
                   google-api-python-client markdown

  2. Create a .env file next to this script with:
       OPENAI_API_KEY=sk-...
       GMAIL_SENDER=you@gmail.com
       GMAIL_RECIPIENT=recipient@gmail.com

  3. Google OAuth Setup (fixes auth errors):
     a) Go to https://console.cloud.google.com/
     b) Create or select a project
     c) Enable the Gmail API:
          APIs & Services → Enable APIs → search "Gmail API" → Enable
     d) Create OAuth credentials:
          APIs & Services → Credentials → Create Credentials → OAuth client ID
          Application type: Desktop app
          Download as credentials.json → place next to this script
     e) Add your Gmail as a Test User:
          APIs & Services → OAuth consent screen → Test users → Add Users
          (Required while app is in "testing" mode — fixes "access blocked" errors)

  4. Place your audio file as 'EarningsCall.wav' next to this script.

  5. Run:
       python meeting_minutes_openai.py

  COMMON GOOGLE AUTH ERRORS AND FIXES:
  - "Access blocked: app not verified"  → Add your Gmail to Test Users (step 3e above)
  - "redirect_uri_mismatch"             → Use Application type: Desktop app (not Web)
  - "invalid_grant" / token expired     → Delete token.json and re-run to re-authenticate
  - "credentials.json not found"        → Download from Cloud Console (step 3d above)
  - "Gmail API not enabled"             → Enable it in Cloud Console (step 3c above)
"""

# ─── CONFIG ──────────────────────────────────────────────────────────────────
AUDIO_FILE        = "EarningsCall.wav"    # path to your .wav recording
OPENAI_MODEL      = "gpt-4o"             # OpenAI model for LLM tasks
WHISPER_MODEL     = "whisper-1"          # OpenAI Whisper model for transcription
CHUNK_LENGTH_MS   = 60_000              # 60-second audio chunks for Whisper
MEETING_DATE      = "2025-01-01"        # override or leave "" to use today
COMPANY_NAME      = "TylerAI"
ORGANIZER_NAME    = "Tyler"
MEETING_LOCATION  = "Zoom"
OUTPUT_DIR        = "meeting_minutes"   # folder where output files are saved
# ─────────────────────────────────────────────────────────────────────────────

import os
import base64
import datetime
import tempfile
from pathlib import Path
from email.mime.text import MIMEText

from dotenv import load_dotenv
load_dotenv()

# Validate OpenAI key early
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError(
        "OPENAI_API_KEY not found.\n"
        "Add it to your .env file: OPENAI_API_KEY=sk-..."
    )


# ── OpenAI client (shared) ────────────────────────────────────────────────────
def get_openai_client():
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("Run: pip install openai")
    return OpenAI(api_key=OPENAI_API_KEY)


# ── Audio transcription (OpenAI Whisper) ──────────────────────────────────────
def transcribe_audio(audio_path: str) -> str:
    """Transcribe a WAV file using OpenAI Whisper (chunked to handle large files)."""
    try:
        from pydub import AudioSegment
        from pydub.utils import make_chunks
    except ImportError:
        raise ImportError("Run: pip install pydub")

    print(f"\n[1/5] Transcribing audio: {audio_path}")
    client  = get_openai_client()
    audio   = AudioSegment.from_file(audio_path, format="wav")
    chunks  = make_chunks(audio, CHUNK_LENGTH_MS)
    full_tx = ""

    with tempfile.TemporaryDirectory() as tmpdir:
        for i, chunk in enumerate(chunks):
            print(f"      chunk {i+1}/{len(chunks)} …")
            chunk_path = os.path.join(tmpdir, f"chunk_{i}.wav")
            chunk.export(chunk_path, format="wav")
            with open(chunk_path, "rb") as f:
                result = client.audio.transcriptions.create(
                    model=WHISPER_MODEL, file=f
                )
            full_tx += result.text + " "

    print(f"      transcription complete ({len(full_tx)} chars)")
    return full_tx.strip()


# ── OpenAI GPT-4o helper ──────────────────────────────────────────────────────
def openai_chat(prompt: str, system: str = "", max_tokens: int = 1500) -> str:
    """Send a prompt to OpenAI GPT-4o and return the response text."""
    client   = get_openai_client()
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.3,     # lower = more consistent/factual output
    )
    return response.choices[0].message.content.strip()


# ── Step 1 – Summarize ────────────────────────────────────────────────────────
def summarize_transcript(transcript: str) -> str:
    print("\n[2/5] Generating summary …")
    system = (
        "You are a highly skilled AI trained in language comprehension and summarization. "
        "Read the meeting transcript and produce a concise abstract paragraph retaining the "
        "most important points. Avoid unnecessary details."
    )
    prompt = f"Please summarize the following meeting transcript:\n\n{transcript}"
    return openai_chat(prompt, system)


# ── Step 2 – Action items ─────────────────────────────────────────────────────
def extract_action_items(transcript: str) -> str:
    print("      Extracting action items …")
    system = "You are an expert at extracting actionable tasks from meeting transcripts."
    prompt = (
        "From the following meeting transcript, extract all action items. "
        "Return them as a bullet list, one per line, starting each with '- '.\n\n"
        f"{transcript}"
    )
    return openai_chat(prompt, system)


# ── Step 3 – Sentiment ────────────────────────────────────────────────────────
def analyze_sentiment(transcript: str) -> str:
    print("      Analyzing sentiment …")
    system = (
        "You are an AI with expertise in language and emotion analysis. "
        "Analyze the overall tone and sentiment of the text."
    )
    prompt = (
        "Analyze the sentiment of the following meeting transcript. "
        "Indicate whether it is positive, negative, or neutral, and briefly explain why.\n\n"
        f"{transcript}"
    )
    return openai_chat(prompt, system)


# ── Step 4 – Write meeting minutes ────────────────────────────────────────────
def write_meeting_minutes(summary: str, action_items: str, sentiment: str) -> str:
    print("\n[3/5] Writing meeting minutes document …")
    today  = MEETING_DATE or datetime.date.today().isoformat()
    system = (
        "You are a skilled writer with a talent for crafting clear, concise, "
        "and well-formatted meeting minutes in Markdown."
    )
    prompt = f"""Write professional meeting minutes as a Markdown document using these details:

Date: {today}
Company: {COMPANY_NAME}
Organizer: {ORGANIZER_NAME}
Location: {MEETING_LOCATION}
Attendees: (create a realistic list of 5–6 attendees based on the summary)

SUMMARY:
{summary}

ACTION ITEMS:
{action_items}

SENTIMENT ANALYSIS:
{sentiment}

Structure the document with sections: Overview, Attendees, Discussion Summary, Action Items, Sentiment, Next Steps.
"""
    return openai_chat(prompt, system, max_tokens=2000)


# ── Save files ────────────────────────────────────────────────────────────────
def save_outputs(summary: str, action_items: str, sentiment: str, minutes: str):
    print(f"\n[4/5] Saving files to ./{OUTPUT_DIR}/ …")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    files = {
        "summary.txt":        summary,
        "action_items.txt":   action_items,
        "sentiment.txt":      sentiment,
        "meeting_minutes.md": minutes,
    }
    for fname, content in files.items():
        path = os.path.join(OUTPUT_DIR, fname)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"      saved → {path}")


# ── Google OAuth (fixed) ──────────────────────────────────────────────────────
def authenticate_gmail():
    """
    Authenticate with Gmail using OAuth 2.0.

    Uses manual copy-paste auth (no local server needed) — works on:
      - Remote servers / VMs
      - WSL (Windows Subsystem for Linux)
      - Headless / no-browser environments
      - Any machine where localhost redirect fails

    Flow:
      1. Script prints a Google URL
      2. You open it in ANY browser (phone, another PC, etc.)
      3. You approve access → Google shows a code
      4. You paste that code back into the terminal
    """
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
    except ImportError:
        raise ImportError(
            "Run: pip install google-auth google-auth-oauthlib google-api-python-client"
        )

    SCOPES           = ["https://www.googleapis.com/auth/gmail.compose"]
    script_dir       = Path(__file__).parent
    token_path       = script_dir / "token.json"
    credentials_path = script_dir / "credentials.json"

    if not credentials_path.exists():
        raise FileNotFoundError(
            "\n❌ credentials.json not found!\n\n"
            "To fix this:\n"
            "  1. Go to https://console.cloud.google.com/\n"
            "  2. APIs & Services → Credentials → Create Credentials → OAuth client ID\n"
            "  3. Application type: Desktop app  ← IMPORTANT (not Web)\n"
            "  4. Download → rename to credentials.json → place next to this script\n"
        )

    creds = None

    # ── Load cached token ──────────────────────────────────────────────────
    if token_path.exists():
        try:
            creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
        except Exception:
            print("      ⚠  token.json is corrupt — will re-authenticate.")
            token_path.unlink(missing_ok=True)
            creds = None

    # ── Refresh if expired ─────────────────────────────────────────────────
    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            print("      🔄 Token refreshed successfully.")
        except Exception as e:
            print(f"      ⚠  Token refresh failed ({e}) — re-authenticating …")
            token_path.unlink(missing_ok=True)
            creds = None

    # ── Manual copy-paste OAuth (no localhost server) ──────────────────────
    if not creds or not creds.valid:
        flow = InstalledAppFlow.from_client_secrets_file(
            str(credentials_path), SCOPES
        )

        # Build the auth URL manually with OOB-style redirect
        flow.redirect_uri = "urn:ietf:wg:oauth:2.0:oob"
        auth_url, _ = flow.authorization_url(
            prompt="consent",
            access_type="offline",
        )

        print("\n" + "=" * 60)
        print("  GOOGLE AUTHENTICATION REQUIRED")
        print("=" * 60)
        print("\n  1. Open this URL in any browser (phone/PC/any device):\n")
        print(f"     {auth_url}\n")
        print("  2. Sign in with your Gmail account and click Allow")
        print("  3. Google will show a code — copy it")
        print("  4. Paste the code below and press Enter\n")
        print("=" * 60)

        code = input("  Paste the authorization code here: ").strip()

        try:
            flow.fetch_token(code=code)
            creds = flow.credentials
        except Exception as e:
            raise RuntimeError(
                f"\n❌ Authorization failed: {e}\n\n"
                "Common fixes:\n"
                "  • Code expired  → Re-run the script and use the new URL immediately\n"
                "  • Wrong account → Make sure you log in with the Gmail in GMAIL_SENDER\n"
                "  • Access blocked → Cloud Console → OAuth consent screen → Test users → add your Gmail\n"
            ) from e

        # Save token so you only need to do this once
        with open(token_path, "w") as f:
            f.write(creds.to_json())
        print("\n      💾 Token saved — you won't need to do this again unless it expires.")

    try:
        service = build("gmail", "v1", credentials=creds)
        # Quick sanity check — verifies Gmail API is enabled
        service.users().getProfile(userId="me").execute()
        return service
    except Exception as e:
        err_str = str(e)
        if "accessNotConfigured" in err_str or "API has not been used" in err_str:
            raise RuntimeError(
                "\n❌ Gmail API is not enabled!\n\n"
                "To fix this:\n"
                "  1. Go to https://console.cloud.google.com/\n"
                "  2. APIs & Services → Enable APIs & Services\n"
                "  3. Search for 'Gmail API' and click Enable\n"
            ) from e
        raise


# ── Gmail draft ───────────────────────────────────────────────────────────────
def create_gmail_draft(minutes_markdown: str):
    print("\n[5/5] Creating Gmail draft …")
    sender    = os.getenv("GMAIL_SENDER")
    recipient = os.getenv("GMAIL_RECIPIENT")

    if not sender or not recipient:
        print(
            "      ⚠  GMAIL_SENDER / GMAIL_RECIPIENT not set in .env — skipping Gmail draft.\n"
            "         Add them to .env:\n"
            "           GMAIL_SENDER=you@gmail.com\n"
            "           GMAIL_RECIPIENT=recipient@gmail.com"
        )
        return

    try:
        import markdown as md_lib
    except ImportError:
        raise ImportError("Run: pip install markdown")

    HTML_TEMPLATE = """<!DOCTYPE html>
<html><head><meta charset="UTF-8"></head>
<body style="font-family:sans-serif;max-width:700px;margin:auto;padding:20px">
{body}
</body></html>"""

    html_body = HTML_TEMPLATE.format(
        body=md_lib.Markdown(
            extensions=["tables", "fenced_code", "nl2br"]
        ).convert(minutes_markdown)
    )

    msg            = MIMEText(html_body, "html", "utf-8")
    msg["To"]      = recipient
    msg["From"]    = sender
    msg["Subject"] = (
        f"Meeting Minutes - {COMPANY_NAME} ({datetime.date.today()})"
    )
    encoded = base64.urlsafe_b64encode(msg.as_bytes()).decode()

    try:
        service = authenticate_gmail()
        draft   = service.users().drafts().create(
            userId="me", body={"message": {"raw": encoded}}
        ).execute()
        print(f"      ✅ Draft created! id={draft['id']}")
        print(f"         Open Gmail → Drafts to review and send.")
    except Exception as e:
        print(f"      ❌ Gmail draft failed: {e}")


# ── Main pipeline ─────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Meeting Minutes Pipeline  (OpenAI GPT-4o + Whisper)")
    print("=" * 60)

    script_dir = Path(__file__).parent
    audio_path = str(script_dir / AUDIO_FILE)

    if not os.path.exists(audio_path):
        raise FileNotFoundError(
            f"\n❌ Audio file not found: {audio_path}\n"
            f"   Place your WAV file next to this script and set AUDIO_FILE at the top."
        )

    # 1. Transcribe audio
    transcript = transcribe_audio(audio_path)

    # 2. Extract insights via GPT-4o
    summary      = summarize_transcript(transcript)
    action_items = extract_action_items(transcript)
    sentiment    = analyze_sentiment(transcript)

    # 3. Write full meeting minutes
    minutes = write_meeting_minutes(summary, action_items, sentiment)

    # 4. Save all outputs to disk
    save_outputs(summary, action_items, sentiment, minutes)

    # 5. Create Gmail draft
    create_gmail_draft(minutes)

    print("\n✅ Done! Check the meeting_minutes/ folder for output files.")
    print("=" * 60)


if __name__ == "__main__":
    main()