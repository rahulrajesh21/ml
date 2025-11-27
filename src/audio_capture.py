"""
Audio capture module for real-time meeting transcription.
Handles live audio input from microphone or system audio.
"""

import pyaudio
import wave
import threading
import queue
import numpy as np
from typing import Optional, Callable
import time


class AudioCapture:
    """
    Captures live audio from microphone and provides it in chunks for real-time processing.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_duration: float = 2.0,  # Duration of each audio chunk in seconds
        channels: int = 1,
        device_index: Optional[int] = None
    ):
        """
        Initialize the audio capture.
        
        Args:
            sample_rate: Audio sample rate (16000 is optimal for Whisper)
            chunk_duration: Duration of each audio chunk in seconds
            channels: Number of audio channels (1 for mono, 2 for stereo)
            device_index: Optional specific audio device index to use
        """
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.channels = channels
        self.device_index = device_index
        
        # Calculate chunk size in frames
        self.chunk_size = int(sample_rate * chunk_duration)
        self.format = pyaudio.paInt16
        
        self.audio = pyaudio.PyAudio()
        self.stream: Optional[pyaudio.Stream] = None
        self.is_recording = False
        self.audio_queue = queue.Queue()
        self.recording_thread: Optional[threading.Thread] = None
        
    def list_audio_devices(self) -> list[dict]:
        """
        List all available audio input devices.
        
        Returns:
            List of dictionaries containing device information
        """
        devices = []
        for i in range(self.audio.get_device_count()):
            device_info = self.audio.get_device_info_by_index(i)
            if device_info['maxInputChannels'] > 0:  # Only input devices
                devices.append({
                    'index': i,
                    'name': device_info['name'],
                    'sample_rate': int(device_info['defaultSampleRate']),
                    'channels': device_info['maxInputChannels']
                })
        return devices
    
    def start_recording(self, callback: Optional[Callable] = None):
        """
        Start capturing audio from the microphone.
        
        Args:
            callback: Optional callback function to call with each audio chunk
        """
        if self.is_recording:
            print("Already recording!")
            return
        
        self.is_recording = True
        
        # Open audio stream
        self.stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            input_device_index=self.device_index,
            frames_per_buffer=1024,
            stream_callback=None
        )
        
        # Start recording thread
        self.recording_thread = threading.Thread(
            target=self._record_audio,
            args=(callback,),
            daemon=True
        )
        self.recording_thread.start()
        
        print(f"Recording started at {self.sample_rate}Hz...")
    
    def _record_audio(self, callback: Optional[Callable]):
        """
        Internal method to continuously record audio chunks.
        
        Args:
            callback: Optional callback function for each chunk
        """
        buffer = []
        frames_collected = 0
        
        while self.is_recording:
            try:
                # Read audio data
                data = self.stream.read(1024, exception_on_overflow=False)
                buffer.append(data)
                frames_collected += 1024
                
                # When we have enough frames for a chunk
                if frames_collected >= self.chunk_size:
                    # Combine buffer into single chunk
                    audio_chunk = b''.join(buffer)
                    
                    # Put in queue
                    self.audio_queue.put(audio_chunk)
                    
                    # Call callback if provided
                    if callback:
                        callback(audio_chunk)
                    
                    # Reset buffer
                    buffer = []
                    frames_collected = 0
                    
            except Exception as e:
                print(f"Error recording audio: {e}")
                break
    
    def stop_recording(self):
        """Stop recording audio."""
        if not self.is_recording:
            print("Not recording!")
            return
        
        self.is_recording = False
        
        # Wait for thread to finish
        if self.recording_thread:
            self.recording_thread.join(timeout=2.0)
        
        # Close stream
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        
        print("Recording stopped.")
    
    def get_audio_chunk(self, timeout: float = 1.0) -> Optional[bytes]:
        """
        Get the next audio chunk from the queue.
        
        Args:
            timeout: Timeout in seconds to wait for a chunk
            
        Returns:
            Audio chunk as bytes, or None if timeout
        """
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def save_chunk_to_wav(self, audio_data: bytes, filename: str):
        """
        Save an audio chunk to a WAV file.
        
        Args:
            audio_data: Audio data as bytes
            filename: Output filename
        """
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.audio.get_sample_size(self.format))
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_data)
    
    def cleanup(self):
        """Clean up resources."""
        self.stop_recording()
        self.audio.terminate()
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.cleanup()
        except:
            pass


class AudioBuffer:
    """
    Manages a rolling buffer of audio chunks with overlap for better transcription.
    """
    
    def __init__(self, max_duration: float = 30.0, sample_rate: int = 16000):
        """
        Initialize audio buffer.
        
        Args:
            max_duration: Maximum duration of audio to keep in buffer (seconds)
            sample_rate: Audio sample rate
        """
        self.max_duration = max_duration
        self.sample_rate = sample_rate
        self.max_frames = int(max_duration * sample_rate)
        self.buffer = []
        self.total_frames = 0
    
    def add_chunk(self, audio_chunk: bytes):
        """
        Add an audio chunk to the buffer.
        
        Args:
            audio_chunk: Audio data as bytes
        """
        # Convert bytes to numpy array
        audio_array = np.frombuffer(audio_chunk, dtype=np.int16)
        self.buffer.append(audio_array)
        self.total_frames += len(audio_array)
        
        # Remove old chunks if buffer is too long
        while self.total_frames > self.max_frames and len(self.buffer) > 1:
            removed = self.buffer.pop(0)
            self.total_frames -= len(removed)
    
    def get_audio(self) -> np.ndarray:
        """
        Get the complete buffered audio as a numpy array.
        
        Returns:
            Combined audio data as numpy array
        """
        if not self.buffer:
            return np.array([], dtype=np.int16)
        return np.concatenate(self.buffer)
    
    def get_audio_bytes(self) -> bytes:
        """
        Get the complete buffered audio as bytes.
        
        Returns:
            Combined audio data as bytes
        """
        return self.get_audio().tobytes()
    
    def clear(self):
        """Clear the buffer."""
        self.buffer = []
        self.total_frames = 0
    
    def get_duration(self) -> float:
        """
        Get the current duration of buffered audio.
        
        Returns:
            Duration in seconds
        """
        return self.total_frames / self.sample_rate


if __name__ == "__main__":
    # Test the audio capture
    print("Testing Audio Capture...")
    
    capture = AudioCapture(chunk_duration=2.0)
    
    # List available devices
    print("\nAvailable audio devices:")
    devices = capture.list_audio_devices()
    for device in devices:
        print(f"  [{device['index']}] {device['name']} - {device['sample_rate']}Hz")
    
    # Record for 10 seconds
    print("\nRecording for 10 seconds...")
    capture.start_recording()
    
    chunks_received = 0
    start_time = time.time()
    
    while time.time() - start_time < 10:
        chunk = capture.get_audio_chunk(timeout=1.0)
        if chunk:
            chunks_received += 1
            print(f"Received chunk {chunks_received} ({len(chunk)} bytes)")
    
    capture.stop_recording()
    capture.cleanup()
    
    print(f"\nTest complete! Received {chunks_received} audio chunks.")
