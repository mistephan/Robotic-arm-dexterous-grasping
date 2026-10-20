"""Optional voice input (speech-to-text) for the dashboard.

The ASR engine (faster-whisper) is a post-install pip dependency -- see
CLAUDE.md "Dependency install gotchas". Everything here imports it lazily
so the mock test stack never needs it installed.
"""
