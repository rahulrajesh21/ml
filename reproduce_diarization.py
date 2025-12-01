import os
import sys
import numpy as np
import logging
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

# Import live_transcription FIRST to apply torch.load monkey-patch
try:
    import live_transcription
except ImportError:
    pass

import numpy as np

def generate_dummy_audio(filename="test_audio.wav", duration=5):
    """Generate a dummy audio file (sine wave)."""
    try:
        import scipy.io.wavfile
        sample_rate = 16000
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        # Generate a 440 Hz sine wave
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)
        # Convert to int16
        audio_int16 = (audio * 32767).astype(np.int16)
        scipy.io.wavfile.write(filename, sample_rate, audio_int16)
        return filename
    except ImportError:
        logger.error("scipy not installed, cannot generate audio file.")
        return None

def test_imports():
    print("\n--- Testing Imports ---")
    
    # Check PyTorch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Check HF_TOKEN
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        print(f"HF_TOKEN found: {hf_token[:4]}...{hf_token[-4:]}")
    else:
        print("WARNING: HF_TOKEN not found in environment variables!")

    # Check WhisperX
    try:
        import whisperx
        print(f"WhisperX imported successfully. Version: {getattr(whisperx, '__version__', 'unknown')}")
    except ImportError as e:
        print(f"ERROR: WhisperX import failed: {e}")

    # Check Pyannote
    try:
        from pyannote.audio import Pipeline
        print("pyannote.audio Pipeline imported successfully.")
    except ImportError as e:
        print(f"ERROR: pyannote.audio import failed: {e}")

def test_transcriber_init(model_type):
    print(f"\n--- Testing LiveTranscriber ({model_type}) ---")
    try:
        from live_transcription import LiveTranscriber
        
        print(f"Initializing {model_type} with diarization=True...")
        transcriber = LiveTranscriber(
            model_type=model_type,
            model_size="tiny", # Use tiny for speed
            device="cpu",
            compute_type="int8",
            enable_diarization=True
        )
        
        print("Initialization successful!")
        print(f"Diarization enabled: {transcriber.enable_diarization}")
        if transcriber.diarize_model:
            print(f"Diarization model loaded: {type(transcriber.diarize_model)}")
        else:
            print("WARNING: Diarization model is None!")
            
        return transcriber
    except Exception as e:
        print(f"ERROR initializing {model_type}: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    test_imports()
    
    # Generate dummy audio
    audio_file = generate_dummy_audio()
    if not audio_file:
        return

    # Test Faster-Whisper
    fw_transcriber = test_transcriber_init("faster-whisper")
    
    # Test WhisperX
    wx_transcriber = test_transcriber_init("whisperx")
    
    # Clean up
    if os.path.exists(audio_file):
        os.remove(audio_file)

if __name__ == "__main__":
    main()
