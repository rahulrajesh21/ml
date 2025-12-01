import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

try:
    from live_transcription import LiveTranscriber, WHISPERX_AVAILABLE
    print(f"Successfully imported LiveTranscriber. WHISPERX_AVAILABLE={WHISPERX_AVAILABLE}")
except ImportError as e:
    print(f"ImportError: {e}")
except NameError as e:
    print(f"NameError: {e}")
except Exception as e:
    print(f"Error: {e}")
