"""
Real-time transcription engine using Whisper for live audio streaming.
Supports both OpenAI Whisper API and local Whisper models.
"""

import os
import sys
import tempfile
import threading
import queue
from typing import Optional, Callable, Literal
import time
import numpy as np

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Try to import whisper (faster-whisper for real-time performance)
try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False
    print("Warning: faster-whisper not available. Install with: pip install faster-whisper")

# Try to import OpenAI for API-based transcription
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from audio_capture import AudioCapture, AudioBuffer


class LiveTranscriber:
    """
    Real-time transcription engine that processes live audio streams.
    """
    
    def __init__(
        self,
        model_type: Literal["faster-whisper", "openai"] = "faster-whisper",
        model_size: str = "base",
        language: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        device: str = "cpu",
        compute_type: str = "int8"
    ):
        """
        Initialize the live transcriber.
        
        Args:
            model_type: Type of model to use ("faster-whisper" or "openai")
            model_size: Model size for faster-whisper (tiny, base, small, medium, large-v3)
            language: Optional language code (e.g., "en", "es", "fr")
            openai_api_key: OpenAI API key (required if using openai model)
            device: Device to run on ("cpu" or "cuda")
            compute_type: Compute type for faster-whisper ("int8", "float16", "float32")
        """
        self.model_type = model_type
        self.model_size = model_size
        self.language = language
        self.device = device
        
        # Initialize model based on type
        if model_type == "faster-whisper":
            if not FASTER_WHISPER_AVAILABLE:
                raise ImportError("faster-whisper is not installed. Install with: pip install faster-whisper")
            
            print(f"Loading faster-whisper model: {model_size}...")
            self.model = WhisperModel(
                model_size,
                device=device,
                compute_type=compute_type
            )
            print("Model loaded successfully!")
            
        elif model_type == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError("openai package is not installed. Install with: pip install openai")
            
            if not openai_api_key:
                openai_api_key = os.getenv("OPENAI_API_KEY")
                if not openai_api_key:
                    raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable.")
            
            self.client = OpenAI(api_key=openai_api_key)
            print("OpenAI client initialized!")
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Transcription state
        self.is_transcribing = False
        self.transcription_thread: Optional[threading.Thread] = None
        self.transcript_queue = queue.Queue()
        self.full_transcript = []
        
    def transcribe_chunk(self, audio_data, sample_rate: int = 16000) -> str:
        """
        Transcribe a single audio chunk.
        
        Args:
            audio_data: Audio data as bytes or numpy array (float32)
            sample_rate: Audio sample rate
            
        Returns:
            Transcribed text
        """
        if self.model_type == "faster-whisper":
            return self._transcribe_with_faster_whisper(audio_data, sample_rate)
        elif self.model_type == "openai":
            return self._transcribe_with_openai(audio_data, sample_rate)
    
    def _transcribe_with_faster_whisper(self, audio_data, sample_rate: int) -> str:
        """
        Transcribe using faster-whisper (local model).
        
        Args:
            audio_data: Audio data as bytes or numpy array (float32)
            sample_rate: Audio sample rate
            
        Returns:
            Transcribed text
        """
        # Convert to float32 array if needed
        if isinstance(audio_data, bytes):
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        elif isinstance(audio_data, np.ndarray):
            # Already a numpy array - ensure it's float32 and normalized
            if audio_data.dtype != np.float32:
                audio_array = audio_data.astype(np.float32)
            else:
                audio_array = audio_data
            
            # Ensure normalized to [-1, 1] range
            if audio_array.max() > 1.0 or audio_array.min() < -1.0:
                audio_array = np.clip(audio_array, -1.0, 1.0)
        else:
            raise TypeError(f"audio_data must be bytes or numpy array, got {type(audio_data)}")
        
        # Transcribe with voice activity detection to filter silence
        segments, info = self.model.transcribe(
            audio_array,
            language=self.language,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500)
        )
        
        # Combine segments
        text = " ".join([segment.text for segment in segments])
        return text.strip()
    
    def _transcribe_with_openai(self, audio_data: bytes, sample_rate: int) -> str:
        """
        Transcribe using OpenAI Whisper API.
        
        Args:
            audio_data: Audio data as bytes
            sample_rate: Audio sample rate
            
        Returns:
            Transcribed text
        """
        # Save audio to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name
            
            # Write WAV file
            import wave
            with wave.open(temp_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(sample_rate)
                wf.writeframes(audio_data)
        
        try:
            # Transcribe with OpenAI API
            with open(temp_path, 'rb') as audio_file:
                transcript = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language=self.language
                )
            return transcript.text.strip()
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def start_live_transcription(
        self,
        audio_capture: AudioCapture,
        callback: Optional[Callable[[dict], None]] = None
    ):
        """
        Start live transcription from audio capture.
        
        Args:
            audio_capture: AudioCapture instance
            callback: Optional callback function called with each transcription
        """
        if self.is_transcribing:
            print("Already transcribing!")
            return
        
        self.is_transcribing = True
        self.full_transcript = []
        
        # Start transcription thread
        self.transcription_thread = threading.Thread(
            target=self._transcription_loop,
            args=(audio_capture, callback),
            daemon=True
        )
        self.transcription_thread.start()
        
        print("Live transcription started!")
    
    def _transcription_loop(
        self,
        audio_capture: AudioCapture,
        callback: Optional[Callable[[dict], None]]
    ):
        """
        Main transcription loop that processes audio chunks.
        
        Args:
            audio_capture: AudioCapture instance
            callback: Optional callback for each transcription
        """
        chunk_count = 0
        
        while self.is_transcribing:
            # Get next audio chunk
            audio_chunk = audio_capture.get_audio_chunk(timeout=1.0)
            
            if audio_chunk is None:
                continue
            
            chunk_count += 1
            
            try:
                # Transcribe the chunk
                start_time = time.time()
                text = self.transcribe_chunk(audio_chunk, audio_capture.sample_rate)
                elapsed = time.time() - start_time
                
                if text:
                    # Add to full transcript
                    timestamp = time.strftime("%H:%M:%S")
                    transcription_entry = {
                        "timestamp": timestamp,
                        "text": text,
                        "chunk": chunk_count
                    }
                    
                    self.full_transcript.append(transcription_entry)
                    self.transcript_queue.put(transcription_entry)
                    
                    print(f"[{timestamp}] {text} (processed in {elapsed:.2f}s)")
                    
                    # Call callback if provided
                    if callback:
                        callback(transcription_entry)
                        
            except Exception as e:
                print(f"Error transcribing chunk {chunk_count}: {e}")
    
    def stop_transcription(self):
        """Stop live transcription."""
        if not self.is_transcribing:
            print("Not transcribing!")
            return
        
        self.is_transcribing = False
        
        # Wait for thread to finish
        if self.transcription_thread:
            self.transcription_thread.join(timeout=5.0)
        
        print("Live transcription stopped.")
    
    def get_full_transcript(self) -> str:
        """
        Get the complete transcript as a formatted string.
        
        Returns:
            Full transcript with timestamps
        """
        lines = []
        for entry in self.full_transcript:
            lines.append(f"[{entry['timestamp']}] {entry['text']}")
        return "\n".join(lines)
    
    def get_transcript_entries(self) -> list[dict]:
        """
        Get all transcript entries.
        
        Returns:
            List of transcript entries with timestamps
        """
        return self.full_transcript.copy()
    
    def clear_transcript(self):
        """Clear the transcript history."""
        self.full_transcript = []
        # Clear queue
        while not self.transcript_queue.empty():
            try:
                self.transcript_queue.get_nowait()
            except queue.Empty:
                break


if __name__ == "__main__":
    # Test the live transcription
    print("Testing Live Transcription...")
    
    # Initialize audio capture
    capture = AudioCapture(chunk_duration=3.0)
    
    # Initialize transcriber (using faster-whisper for local testing)
    transcriber = LiveTranscriber(
        model_type="faster-whisper",
        model_size="base",
        language="en"
    )
    
    # Start recording
    capture.start_recording()
    
    # Start transcription
    transcriber.start_live_transcription(capture)
    
    print("\nSpeak into your microphone... (recording for 30 seconds)")
    time.sleep(30)
    
    # Stop everything
    transcriber.stop_transcription()
    capture.stop_recording()
    
    # Print full transcript
    print("\n" + "="*50)
    print("FULL TRANSCRIPT:")
    print("="*50)
    print(transcriber.get_full_transcript())
    
    # Cleanup
    capture.cleanup()
