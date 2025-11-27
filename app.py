"""
Gradio-based web interface for live meeting transcription.
Provides real-time transcription with controls and export functionality.
"""

import gradio as gr
import time
import threading
from typing import Optional
import os
import sys
import json
from datetime import datetime
import subprocess
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.audio_capture import AudioCapture
from src.live_transcription import LiveTranscriber


class LiveMeetingApp:
    """
    Application class for managing live meeting transcription with Gradio UI.
    """
    
    def __init__(self):
        self.audio_capture: Optional[AudioCapture] = None
        self.transcriber: Optional[LiveTranscriber] = None
        self.is_running = False
        self.transcript_text = ""
        self.update_lock = threading.Lock()
        self.is_simulating = False
        self.simulation_thread = None
        self.video_audio_path = None
        
    def initialize_components(
        self,
        model_type: str,
        model_size: str,
        language: str,
        device: str
    ):
        """
        Initialize audio capture and transcriber components.
        
        Args:
            model_type: Type of model ("faster-whisper" or "openai")
            model_size: Model size (for faster-whisper)
            language: Language code
            device: Device to use ("cpu" or "cuda")
        """
        # Clean up existing components
        if self.audio_capture:
            try:
                self.audio_capture.cleanup()
            except:
                pass
        
        # Initialize audio capture
        self.audio_capture = AudioCapture(
            sample_rate=16000,
            chunk_duration=3.0,  # 3-second chunks for real-time processing
            channels=1
        )
        
        # Get OpenAI API key from environment if using OpenAI
        openai_api_key = None
        if model_type == "openai":
            openai_api_key = os.getenv("OPENAI_API_KEY")
        
        # Initialize transcriber
        self.transcriber = LiveTranscriber(
            model_type=model_type,
            model_size=model_size if model_type == "faster-whisper" else "whisper-1",
            language=language if language != "auto" else None,
            openai_api_key=openai_api_key,
            device=device
        )
        
    def start_recording(
        self,
        model_type: str,
        model_size: str,
        language: str,
        device: str
    ) -> tuple[str, str]:
        """
        Start live transcription.
        
        Returns:
            Tuple of (status message, initial transcript)
        """
        if self.is_running:
            return "⚠️ Already recording!", self.transcript_text
        
        try:
            # Initialize components
            self.initialize_components(model_type, model_size, language, device)
            
            # Clear previous transcript
            self.transcript_text = ""
            
            # Start recording
            self.audio_capture.start_recording()
            
            # Start transcription with callback
            self.transcriber.start_live_transcription(
                self.audio_capture,
                callback=self.on_transcription_update
            )
            
            self.is_running = True
            
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            status = f"✅ Recording started at {timestamp}\n\n🎤 Speak into your microphone...\n\nTranscription will appear in real-time on the right."
            
            return status, self.transcript_text
            
        except Exception as e:
            return f"❌ Error starting recording: {str(e)}", ""
    
    def stop_recording(self) -> tuple[str, str]:
        """
        Stop live transcription.
        
        Returns:
            Tuple of (status message, final transcript)
        """
        if not self.is_running:
            return "⚠️ Not recording!", self.transcript_text
        
        try:
            # Stop transcription
            if self.transcriber:
                self.transcriber.stop_transcription()
            
            # Stop recording
            if self.audio_capture:
                self.audio_capture.stop_recording()
            
            self.is_running = False
            
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            word_count = len(self.transcript_text.split())
            status = f"⏹️ Recording stopped at {timestamp}\n\n📊 Statistics:\n- Total words: {word_count}\n\n💾 Transcript is ready for export."
            
            return status, self.transcript_text
            
        except Exception as e:
            return f"❌ Error stopping recording: {str(e)}", self.transcript_text
    
    def on_transcription_update(self, entry: dict):
        """
        Callback function for transcription updates.
        
        Args:
            entry: Dictionary with timestamp, text, and chunk number
        """
        with self.update_lock:
            # Add new line to transcript
            new_line = f"[{entry['timestamp']}] {entry['text']}\n\n"
            self.transcript_text += new_line
    
    def get_transcript(self) -> str:
        """
        Get the current transcript text.
        
        Returns:
            Current transcript
        """
        with self.update_lock:
            return self.transcript_text
    
    def export_transcript(self, format_type: str) -> str:
        """
        Export transcript to file.
        
        Args:
            format_type: Export format ("txt", "json", "srt")
            
        Returns:
            Status message with file path
        """
        if not self.transcript_text and not self.transcriber:
            return "❌ No transcript to export. Please record something first."
        
        # Create exports directory if it doesn't exist
        exports_dir = "exports"
        os.makedirs(exports_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format_type == "txt":
            filename = os.path.join(exports_dir, f"transcript_{timestamp}.txt")
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("LIVE MEETING TRANSCRIPT\n")
                f.write("=" * 50 + "\n\n")
                f.write(self.transcript_text)
        
        elif format_type == "json":
            filename = os.path.join(exports_dir, f"transcript_{timestamp}.json")
            entries = self.transcriber.get_transcript_entries() if self.transcriber else []
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump({
                    "export_timestamp": timestamp,
                    "entries": entries,
                    "full_text": self.transcript_text,
                    "word_count": len(self.transcript_text.split())
                }, f, indent=2, ensure_ascii=False)
        
        elif format_type == "srt":
            filename = os.path.join(exports_dir, f"transcript_{timestamp}.srt")
            entries = self.transcriber.get_transcript_entries() if self.transcriber else []
            with open(filename, 'w', encoding='utf-8') as f:
                for i, entry in enumerate(entries, 1):
                    # Simple SRT format
                    f.write(f"{i}\n")
                    f.write(f"{entry['timestamp']},000 --> {entry['timestamp']},000\n")
                    f.write(f"{entry['text']}\n\n")
        
        abs_path = os.path.abspath(filename)
        return f"✅ Transcript exported successfully!\n\n📁 File saved to:\n{abs_path}"
    
    def process_video_upload(self, video_file) -> str:
        """
        Validate video file for live simulation.
        
        Args:
            video_file: Uploaded video file path
            
        Returns:
            Status message
        """
        if not video_file or video_file.strip() == "":
            return "⚠️ Please enter a video file path first."
        
        # Check if file exists
        if not os.path.exists(video_file):
            return f"❌ File not found: {video_file}\n\nPlease enter the full path to your video file."
        
        try:
            from moviepy import VideoFileClip
            
            # Just validate the video file
            video = VideoFileClip(video_file)
            
            # Check if video has audio
            if video.audio is None:
                video.close()
                return "❌ Error: Video file has no audio track."
            
            duration = video.duration
            video.close()
            
            # Store the video path for live simulation
            self.video_audio_path = video_file
            
            return f"✅ Video loaded successfully!\n\n📹 Duration: {duration:.1f} seconds\n🎬 Ready for live simulation\n\n👉 Click 'Start Video Simulation' to begin real-time transcription."
            
        except ImportError as ie:
            return f"❌ Error: moviepy not installed properly.\n\nError details: {str(ie)}\n\nTry: pip install moviepy"
        except Exception as e:
            return f"❌ Error loading video: {str(e)}\n\nMake sure the file path is correct and the video format is supported."
    
    def start_video_simulation(
        self,
        model_type: str,
        model_size: str,
        language: str,
        device: str,
        speed_multiplier: float
    ) -> tuple[str, str]:
        """
        Start video simulation mode - transcribe video audio in chunks.
        
        Args:
            model_type: Type of model
            model_size: Model size
            language: Language code
            device: Device to use
            speed_multiplier: Playback speed (1.0 = real-time, 2.0 = 2x speed)
            
        Returns:
            Tuple of (status message, initial transcript)
        """
        if self.is_simulating or self.is_running:
            return "⚠️ Already running!", self.transcript_text
        
        if not self.video_audio_path or not os.path.exists(self.video_audio_path):
            return "⚠️ Please upload and process a video first.", ""
        
        try:
            # Initialize transcriber
            openai_api_key = None
            if model_type == "openai":
                openai_api_key = os.getenv("OPENAI_API_KEY")
            
            self.transcriber = LiveTranscriber(
                model_type=model_type,
                model_size=model_size if model_type == "faster-whisper" else "whisper-1",
                language=language if language != "auto" else None,
                openai_api_key=openai_api_key,
                device=device
            )
            
            # Clear previous transcript
            self.transcript_text = ""
            self.is_simulating = True
            
            # Start simulation in background thread
            self.simulation_thread = threading.Thread(
                target=self._simulate_video_transcription,
                args=(speed_multiplier,),
                daemon=True
            )
            self.simulation_thread.start()
            
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            status = f"✅ Video simulation started at {timestamp}\n\n🎬 Processing video audio at {speed_multiplier}x speed...\n\nTranscription will appear on the right."
            
            return status, self.transcript_text
            
        except Exception as e:
            self.is_simulating = False
            return f"❌ Error starting simulation: {str(e)}", ""
    
    def _simulate_video_transcription(self, speed_multiplier: float):
        """
        Simulate live video transcription by reading audio in real-time chunks.
        
        Args:
            speed_multiplier: Speed multiplier for simulation
        """
        try:
            from moviepy import VideoFileClip
            
            print(f"Starting video simulation from: {self.video_audio_path}")
            
            # Open video file
            video = VideoFileClip(self.video_audio_path)
            
            if video.audio is None:
                print("Error: Video has no audio")
                self.is_simulating = False
                return
            
            # Get audio properties
            chunk_duration = 3.0  # 3 second chunks
            total_duration = video.duration
            
            print(f"Video duration: {total_duration}s, chunk duration: {chunk_duration}s")
            
            chunk_num = 0
            current_time = 0
            
            while self.is_simulating and current_time < total_duration:
                # Calculate chunk time range
                start_time = current_time
                end_time = min(current_time + chunk_duration, total_duration)
                
                if chunk_num % 10 == 0:  # Log every 10th chunk
                    print(f"Processing chunk {chunk_num}: {start_time:.1f}s - {end_time:.1f}s")
                
                try:
                    # Use ffmpeg to extract audio chunk directly from video
                    target_fps = 16000
                    duration = end_time - start_time
                    
                    cmd = [
                        'ffmpeg',
                        '-ss', str(start_time),
                        '-t', str(duration),
                        '-i', self.video_audio_path,
                        '-vn',
                        '-acodec', 'pcm_s16le',
                        '-ar', str(target_fps),
                        '-ac', '1',
                        '-f', 's16le',
                        '-loglevel', 'error',  # Suppress ffmpeg output
                        '-'
                    ]
                    
                    result = subprocess.run(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        check=True
                    )
                    
                    # Convert bytes to float32 numpy array
                    audio_int16 = np.frombuffer(result.stdout, dtype=np.int16)
                    audio_float = audio_int16.astype(np.float32) / 32768.0
                    
                    # Transcribe chunk
                    try:
                        result = self.transcriber.transcribe_chunk(audio_float, 16000)
                        
                        if result and result.strip():
                            timestamp = time.strftime('%H:%M:%S', time.gmtime(start_time))
                            entry = {
                                'timestamp': timestamp,
                                'text': result,
                                'chunk': chunk_num
                            }
                            self.on_transcription_update(entry)
                            if chunk_num % 10 == 0:  # Log every 10th transcription
                                print(f"[{timestamp}] Transcribed chunk {chunk_num}")
                    except Exception as e:
                        print(f"Error transcribing chunk {chunk_num}: {e}")
                    
                except subprocess.CalledProcessError as e:
                    print(f"FFmpeg error for chunk {chunk_num}: {e.stderr.decode() if e.stderr else str(e)}")
                except Exception as e:
                    print(f"Error processing chunk {chunk_num}: {e}")
                
                # Sleep to simulate real-time playback
                time.sleep(chunk_duration / speed_multiplier)
                
                current_time = end_time
                chunk_num += 1
            
            # Close video
            video.close()
            
            # Simulation complete
            self.is_simulating = False
            print(f"Video simulation complete. Processed {chunk_num} chunks.")
            
        except Exception as e:
            print(f"Error in video simulation: {e}")
            self.is_simulating = False
        finally:
            if 'video' in locals():
                video.close()
    
    def stop_video_simulation(self) -> tuple[str, str]:
        """
        Stop video simulation.
        
        Returns:
            Tuple of (status message, final transcript)
        """
        if not self.is_simulating:
            return "⚠️ Not simulating!", self.transcript_text
        
        self.is_simulating = False
        
        # Wait for thread to finish
        if self.simulation_thread:
            self.simulation_thread.join(timeout=2.0)
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        word_count = len(self.transcript_text.split())
        status = f"⏹️ Simulation stopped at {timestamp}\n\n📊 Statistics:\n- Total words: {word_count}\n\n💾 Transcript is ready for export."
        
        return status, self.transcript_text
    
    def list_audio_devices(self) -> str:
        """
        List available audio input devices.
        
        Returns:
            Formatted string of devices
        """
        try:
            temp_capture = AudioCapture()
            devices = temp_capture.list_audio_devices()
            temp_capture.cleanup()
            
            if not devices:
                return "❌ No audio input devices found."
            
            result = "🔊 Available Audio Input Devices:\n\n"
            for device in devices:
                result += f"[{device['index']}] {device['name']}\n"
                result += f"    📊 Sample Rate: {device['sample_rate']}Hz\n"
                result += f"    🎵 Channels: {device['channels']}\n\n"
            
            return result
        except Exception as e:
            return f"❌ Error listing devices: {str(e)}"


def create_gradio_interface():
    """
    Create and configure the Gradio interface.
    
    Returns:
        Gradio Blocks interface
    """
    app = LiveMeetingApp()
    
    with gr.Blocks(title="Live Meeting Transcription", theme=gr.themes.Soft()) as interface:
        gr.Markdown(
            """
            # 🎙️ Live Meeting Transcription System
            
            Real-time transcription for meetings, lectures, interviews, and conversations using AI.
            """
        )
        
        with gr.Tabs():
            # Tab 1: Live Recording
            with gr.Tab("🎤 Live Recording"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### ⚙️ Configuration")
                        
                        model_type = gr.Radio(
                            choices=["faster-whisper", "openai"],
                            value="faster-whisper",
                            label="🤖 Model Type",
                            info="faster-whisper runs locally (free), OpenAI uses API (requires key)"
                        )
                        
                        model_size = gr.Dropdown(
                            choices=["tiny", "base", "small", "medium", "large-v3"],
                            value="base",
                            label="📏 Model Size (faster-whisper only)",
                            info="Larger = more accurate but slower"
                        )
                        
                        language = gr.Dropdown(
                            choices=["auto", "en", "es", "fr", "de", "it", "pt", "ru", "ja", "zh", "hi", "ar"],
                            value="en",
                            label="🌍 Language",
                            info="Select 'auto' for automatic detection"
                        )
                        
                        device = gr.Radio(
                            choices=["cpu", "cuda"],
                            value="cpu",
                            label="💻 Device",
                            info="Use CUDA if you have a compatible NVIDIA GPU"
                        )
                        
                        gr.Markdown("---")
                        gr.Markdown("### 🎛️ Controls")
                        
                        start_btn = gr.Button("▶️ Start Recording", variant="primary", size="lg")
                        stop_btn = gr.Button("⏹️ Stop Recording", variant="stop", size="lg")
                        
                        status_box = gr.Textbox(
                            label="📊 Status",
                            value="Ready to start recording...\n\nClick 'Start Recording' to begin.",
                            interactive=False,
                            lines=6
                        )
                        
                        gr.Markdown("---")
                        gr.Markdown("### 🔊 Audio Devices")
                        devices_btn = gr.Button("🔍 List Audio Devices", size="sm")
                        devices_info = gr.Textbox(
                            label="Device Information",
                            lines=8,
                            interactive=False
                        )
                    
                    with gr.Column(scale=2):
                        gr.Markdown("### 📝 Live Transcript")
                        
                        transcript_box = gr.Textbox(
                            label="",
                            placeholder="⏳ Transcript will appear here in real-time as you speak...\n\nMake sure your microphone is connected and working.",
                            lines=35,
                            max_lines=35,
                            interactive=False,
                            show_copy_button=True,
                            autoscroll=True
                        )
                        
                        # Auto-update transcript while recording
                        def update_transcript():
                            while True:
                                if app.is_running:
                                    yield app.get_transcript()
                                time.sleep(1)
                        
                        # Timer to update transcript
                        interface.load(
                            update_transcript,
                            outputs=transcript_box,
                            every=1
                        )
            
            # Tab 2: Video Simulation
            with gr.Tab("🎬 Video Simulation"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### ⚙️ Configuration")
                        
                        video_model_type = gr.Radio(
                            choices=["faster-whisper", "openai"],
                            value="faster-whisper",
                            label="🤖 Model Type",
                            info="faster-whisper runs locally (free), OpenAI uses API (requires key)"
                        )
                        
                        video_model_size = gr.Dropdown(
                            choices=["tiny", "base", "small", "medium", "large-v3"],
                            value="base",
                            label="📏 Model Size (faster-whisper only)",
                            info="Larger = more accurate but slower"
                        )
                        
                        video_language = gr.Dropdown(
                            choices=["auto", "en", "es", "fr", "de", "it", "pt", "ru", "ja", "zh", "hi", "ar"],
                            value="en",
                            label="🌍 Language",
                            info="Select 'auto' for automatic detection"
                        )
                        
                        video_device = gr.Radio(
                            choices=["cpu", "cuda"],
                            value="cpu",
                            label="💻 Device",
                            info="Use CUDA if you have a compatible NVIDIA GPU"
                        )
                        
                        speed_multiplier = gr.Slider(
                            minimum=0.5,
                            maximum=5.0,
                            value=1.0,
                            step=0.5,
                            label="⚡ Playback Speed",
                            info="1.0 = real-time, 2.0 = 2x speed (faster processing)"
                        )
                        
                gr.Markdown("---")
                gr.Markdown("### 📹 Video Upload")
                
                gr.Markdown(
                    """
                    Enter the path to your video file to simulate a live meeting.  
                    The video will be processed in real-time as if it were a live stream.
                    """
                )
                
                video_upload = gr.Textbox(
                    label="Video File Path",
                    placeholder="/path/to/your/video.mp4",
                    lines=1
                )
                
                process_video_btn = gr.Button("📂 Load Video", size="sm")
                video_status = gr.Textbox(
                    label="Video Status",
                    interactive=False,
                    lines=6
                )
                
                gr.Markdown("---")
                gr.Markdown("### 🎛️ Controls")
                
                start_sim_btn = gr.Button("▶️ Start Video Simulation", variant="primary", size="lg")
                stop_sim_btn = gr.Button("⏹️ Stop Simulation", variant="stop", size="lg")
                
                sim_status_box = gr.Textbox(
                    label="📊 Status",
                    value="Upload a video to begin...",
                    interactive=False,
                    lines=6
                )
            
            with gr.Column(scale=2):
                gr.Markdown("### 📝 Video Transcript")
                
                video_transcript_box = gr.Textbox(
                    label="",
                    placeholder="⏳ Transcript will appear here as the video is processed...\n\nUpload a video file and click 'Process Video' to begin.",
                    lines=35,
                    max_lines=35,
                    interactive=False,
                    show_copy_button=True,
                    autoscroll=True
                )
                
                # Auto-update transcript while simulating
                def update_video_transcript():
                    while True:
                        if app.is_simulating:
                            yield app.get_transcript()
                        time.sleep(1)
                
                # Timer to update video transcript
                interface.load(
                    update_video_transcript,
                    outputs=video_transcript_box,
                    every=1
                )
            
            # Tab 3: Export & Settings
            with gr.Tab("💾 Export & Settings"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 💾 Export Transcript")
                        
                        export_format = gr.Radio(
                            choices=["txt", "json", "srt"],
                            value="txt",
                            label="📄 Export Format",
                            info="TXT: plain text, JSON: structured data, SRT: subtitle format"
                        )
                        
                        export_btn = gr.Button("📥 Export Transcript", size="lg", variant="primary")
                        export_file = gr.Textbox(
                            label="Export Status",
                            interactive=False,
                            lines=3
                        )
        
        # Event handlers - Live Recording
        start_btn.click(
            fn=app.start_recording,
            inputs=[model_type, model_size, language, device],
            outputs=[status_box, transcript_box]
        )
        
        stop_btn.click(
            fn=app.stop_recording,
            outputs=[status_box, transcript_box]
        )
        
        devices_btn.click(
            fn=app.list_audio_devices,
            outputs=[devices_info]
        )
        
        # Event handlers - Video Simulation
        process_video_btn.click(
            fn=app.process_video_upload,
            inputs=[video_upload],
            outputs=[video_status]
        )
        
        start_sim_btn.click(
            fn=app.start_video_simulation,
            inputs=[video_model_type, video_model_size, video_language, video_device, speed_multiplier],
            outputs=[sim_status_box, video_transcript_box]
        )
        
        stop_sim_btn.click(
            fn=app.stop_video_simulation,
            outputs=[sim_status_box, video_transcript_box]
        )
        
        # Event handlers - Export
        export_btn.click(
            fn=app.export_transcript,
            inputs=[export_format],
            outputs=[export_file]
        )
        
        gr.Markdown(
            """
            ---
            
            ## 📖 How to Use
            
            ### 🎤 Live Recording Mode
            
            1. **Configure Settings**: Choose your model type, size, and language
            2. **Check Audio**: Click "List Audio Devices" to verify your microphone
            3. **Start Recording**: Click the start button and begin speaking
            4. **Watch Transcript**: Real-time transcription appears on the right (updates every 3 seconds)
            5. **Stop Recording**: Click stop when finished
            6. **Export**: Go to Export & Settings tab to download your transcript
            
            ### 🎬 Video Simulation Mode
            
            1. **Upload Video**: Click "Upload Video File" and select a video (MP4, AVI, MOV, etc.)
            2. **Load Video**: Click "Load Video" to validate the file
            3. **Configure Settings**: Choose your transcription model and settings
            4. **Adjust Speed**: Set playback speed (1.0 = real-time, 2.0 = 2x faster)
            5. **Start Simulation**: Click "Start Video Simulation" to begin
            6. **Watch Live**: Transcript appears in real-time as the video plays
            7. **Export**: Save the transcript from the Export & Settings tab
            
            **Note**: The video is processed in real-time chunks (like a live stream), not pre-processed!
            
            ## 💡 Tips for Best Results
            
            - **Model Selection**:
              - `tiny/base`: Fast, good for real-time (recommended)
              - `small/medium`: Better accuracy, slightly slower
              - `large-v3`: Best accuracy, slowest processing
            
            - **Audio Quality**:
              - Use a good microphone for better accuracy
              - Minimize background noise
              - Speak clearly and at a moderate pace
              - Keep microphone at consistent distance
            
            - **Video Processing**:
              - Supports most video formats (MP4, AVI, MOV, MKV, etc.)
              - Audio is extracted in real-time during simulation
              - Use speed multiplier to process faster than real-time
              - Video is processed like a live stream (no pre-processing needed)
              - 3-second chunks are transcribed on-the-fly
            
            - **Performance**:
              - CPU mode works well for base/small models
              - GPU (CUDA) recommended for medium/large models
              - Video simulation can be faster than real-time with speed multiplier
            
            ## ⚙️ Requirements
            
            - **faster-whisper**: Runs locally, free, no API key needed
            - **OpenAI API**: Requires `OPENAI_API_KEY` environment variable
            - **Video Processing**: Requires moviepy (install: `pip install moviepy`)
            
            """
        )
    
    return interface


if __name__ == "__main__":
    print("Starting Live Meeting Transcription System...")
    interface = create_gradio_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        debug=False
    )
