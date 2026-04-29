# Entire-meeting-minutes-workflow-with-AI
No more manually writing summaries after calls. This Python pipeline handles everything end-to-end:

🎙️ Transcribes your meeting recording (OpenAI Whisper)
📝 Generates a structured summary (GPT-4o)
✅ Extracts action items automatically
📊 Analyzes the meeting sentiment
📧 Creates a Gmail draft — ready to send

You drop in a .wav file, add your API key, and it outputs polished meeting minutes + sends them as a formatted HTML email.

The stack is intentionally simple:
→ OpenAI API (Whisper + GPT-4o)
→ Google Gmail API (OAuth 2.0)
→ Pure Python, single file, no frameworks

One thing I spent time getting right: the Google OAuth flow. The classic "localhost refused to connect" error breaks everyone running this on a server or WSL. Fixed it with a manual copy-paste auth method — open the URL on any device, paste the code back. Works anywhere.
