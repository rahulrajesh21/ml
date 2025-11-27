# Configuration file for live meeting transcription

# Default Settings
DEFAULT_MODEL_TYPE = "faster-whisper"
DEFAULT_MODEL_SIZE = "base"
DEFAULT_LANGUAGE = "en"
DEFAULT_DEVICE = "cpu"

# Audio Settings
SAMPLE_RATE = 16000  # Hz - Optimal for Whisper models
CHUNK_DURATION = 3.0  # seconds - Duration of each audio chunk
AUDIO_CHANNELS = 1  # Mono audio

# Transcription Settings
VAD_FILTER = True  # Voice Activity Detection - filters out silence
MIN_SILENCE_DURATION = 500  # ms - Minimum silence duration for VAD
BEAM_SIZE = 5  # Beam size for transcription (higher = more accurate but slower)

# Server Settings
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 7860
SHARE_GRADIO = False  # Set to True to create public Gradio link
DEBUG_MODE = True

# Export Settings
EXPORT_DIRECTORY = "exports"
EXPORT_FORMATS = ["txt", "json", "srt"]

# Model Options
AVAILABLE_MODELS = {
    "faster-whisper": ["tiny", "base", "small", "medium", "large-v3"],
    "openai": ["whisper-1"]
}

AVAILABLE_LANGUAGES = {
    "auto": "Auto Detect",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "ja": "Japanese",
    "zh": "Chinese",
    "hi": "Hindi",
    "ar": "Arabic",
    "ko": "Korean"
}

# Performance Settings
CPU_THREADS = 4  # Number of CPU threads for faster-whisper
COMPUTE_TYPE = "int8"  # "int8", "float16", or "float32"

# UI Settings
TRANSCRIPT_MAX_LINES = 35
STATUS_LINES = 6
DEVICE_INFO_LINES = 10
