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
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.audio_capture import AudioCapture
from src.live_transcription import LiveTranscriber
from src.text_analysis import TextAnalyzer


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
        self.video_audio_path = None
        self.is_transcribing_video = False
        self.text_analyzer = None


        
    def initialize_components(
        self,
        model_type: str,
        model_size: str,
        language: str,
        device: str,
        enable_diarization: bool = False
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
        
        # Initialize text analyzer (lazy load or background load could be better, but simple for now)
        if not self.text_analyzer:
            print("Initializing Text Analyzer...")
            self.text_analyzer = TextAnalyzer(device=device)

        
        # Get OpenAI API key from environment if using OpenAI
        openai_api_key = None
        if model_type == "openai":
            openai_api_key = os.getenv("OPENAI_API_KEY")
        
        # Get HuggingFace token for WhisperX diarization
        hf_token = None
        if model_type == "whisperx":
            hf_token = os.getenv("HF_TOKEN")
        
        # Initialize transcriber with appropriate model size
        # OpenAI uses "whisper-1", WhisperX and faster-whisper use the selected size
        actual_model_size = model_size
        if model_type == "openai":
            actual_model_size = "whisper-1"
        
        self.transcriber = LiveTranscriber(
            model_type=model_type,
            model_size=actual_model_size,
            language=language if language != "auto" else None,
            openai_api_key=openai_api_key,
            hf_token=hf_token,
            device=device,
            enable_diarization=enable_diarization
        )
        
    def start_recording(
        self,
        model_type: str,
        model_size: str,
        language: str,
        device: str,
        enable_diarization: bool = False
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
            self.initialize_components(model_type, model_size, language, device, enable_diarization)
            
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
    
    def transcribe_video(
        self,
        model_type: str,
        model_size: str,
        language: str,
        device: str,
        enable_diarization: bool = False
    ) -> tuple[str, str]:
        """
        Transcribe the entire video file at once.
        
        Args:
            model_type: Type of model
            model_size: Model size
            language: Language code
            device: Device to use
            enable_diarization: Enable speaker diarization
            
        Returns:
            Tuple of (status message, transcript)
        """
        if self.is_transcribing_video:
            return "⚠️ Already transcribing a video!", self.transcript_text
        
        if not self.video_audio_path or not os.path.exists(self.video_audio_path):
            return "⚠️ Please upload and process a video first.", ""
        
        try:
            self.is_transcribing_video = True
            
            # Initialize transcriber
            openai_api_key = None
            if model_type == "openai":
                openai_api_key = os.getenv("OPENAI_API_KEY")
            
            hf_token = None
            if model_type == "whisperx":
                hf_token = os.getenv("HF_TOKEN")
            
            actual_model_size = model_size
            if model_type == "openai":
                actual_model_size = "whisper-1"
            
            self.transcriber = LiveTranscriber(
                model_type=model_type,
                model_size=actual_model_size,
                language=language if language != "auto" else None,
                openai_api_key=openai_api_key,
                hf_token=hf_token,
                device=device,
                enable_diarization=enable_diarization
            )
            
            self.transcript_text = ""
            
            # Extract full audio
            print(f"Extracting audio from: {self.video_audio_path}")
            target_fps = 16000
            cmd = [
                'ffmpeg',
                '-i', self.video_audio_path,
                '-vn',
                '-acodec', 'pcm_s16le',
                '-ar', str(target_fps),
                '-ac', '1',
                '-f', 's16le',
                '-loglevel', 'error',
                '-'
            ]
            
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
            
            # Transcribe full audio
            print("Starting batch transcription...")
            segments_generator = self.transcriber.transcribe_full_audio(result.stdout, target_fps)
            
            # Format transcript incrementally
            full_text = ""
            for segment in segments_generator:
                start_time = time.strftime('%H:%M:%S', time.gmtime(segment['start']))
                text = segment['text']
                speaker = segment.get('speaker')
                
                if speaker:
                    line = f"[{start_time}] [{speaker}] {text}\n\n"
                else:
                    line = f"[{start_time}] {text}\n\n"
                
                full_text += line
                
                # Update shared state for UI
                self.transcript_text = full_text
            
            self.is_transcribing_video = False
            
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            word_count = len(self.transcript_text.split())
            status = f"✅ Transcription complete at {timestamp}\n\n📊 Statistics:\n- Total words: {word_count}\n\n💾 Transcript is ready for export."
            
            return status, self.transcript_text
            
        except Exception as e:
            self.is_transcribing_video = False
            return f"❌ Error transcribing video: {str(e)}", ""

    def analyze_transcript(self) -> tuple[str, str, float]:
        """
        Analyze the current transcript for sentiment.
        
        Returns:
            Tuple of (status, sentiment_label, sentiment_score)
        """
        if not self.transcript_text:
            return "⚠️ No transcript to analyze!", "N/A", 0.0
            
        if not self.text_analyzer:
            return "⚠️ Text Analyzer not initialized!", "ERROR", 0.0
            
        try:
            print("Analyzing transcript sentiment...")
            result = self.text_analyzer.analyze_sentiment(self.transcript_text)
            
            label = result['label']
            score = result['score']
            
            status = f"✅ Analysis complete.\nSentiment: {label} ({score:.2f})"
            return status, label, score
            
        except Exception as e:
            return f"❌ Error analyzing transcript: {str(e)}", "ERROR", 0.0


    
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
    
    # Create Blocks with optional theme support. Some gradio versions
    # do not accept a `theme` keyword on BlockContext.__init__.
    # Detect support and only pass the argument when available.
    try:
        theme_obj = None
        # gr.themes may not exist on older gradio versions
        if hasattr(gr, "themes"):
            try:
                theme_obj = gr.themes.Soft()
            except Exception:
                theme_obj = None
        blocks_kwargs = {"title": "Live Meeting Transcription"}
        if theme_obj is not None:
            blocks_kwargs["theme"] = theme_obj
        with gr.Blocks(**blocks_kwargs) as interface:
            pass
    except TypeError:
        # Fallback for gradio versions that don't support theme
        with gr.Blocks(title="Live Meeting Transcription") as interface:
            pass
    # Re-enter the Blocks context properly below (we'll create the real context now)
    if "interface" in locals():
        interface.close()
    # Now open the Blocks context without precreating theme (use dynamic kwargs again)
    blocks_kwargs = {"title": "Live Meeting Transcription"}
    if hasattr(gr, "themes"):
        try:
            blocks_kwargs["theme"] = gr.themes.Soft()
        except Exception:
            pass
    with gr.Blocks(**blocks_kwargs) as interface:
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
                            choices=["faster-whisper", "whisperx", "openai"],
                            value="faster-whisper",
                            label="🤖 Model Type",
                            info="faster-whisper: fast & reliable | whisperx: 70x faster + diarization (requires: pip install whisperx) | openai: cloud API"
                        )
                        
                        model_size = gr.Dropdown(
                            choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
                            value="base",
                            label="📏 Model Size",
                            info="Larger = more accurate but slower"
                        )
                        
                        language = gr.Dropdown(
                            choices=["auto", "en", "es", "fr", "de", "it", "pt", "ru", "ja", "zh", "hi", "ar"],
                            value="en",
                            label="🌍 Language",
                            info="Select 'auto' for automatic detection"
                        )
                        
                        device = gr.Radio(
                            choices=["cpu", "cuda", "mps"],
                            value="cpu",
                            label="💻 Device",
                            info="cpu: All models | cuda: All models (NVIDIA GPU) | mps: WhisperX only (Apple M1/M2/M3)"
                        )
                        
                        enable_diarization = gr.Checkbox(
                            value=False,
                            label="👥 Enable Speaker Diarization (WhisperX only)",
                            info="Identifies different speakers (requires HF_TOKEN env variable)"
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
            
            # Tab 2: Video Transcription
            with gr.Tab("🎬 Video Transcription"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### ⚙️ Configuration")
                        
                        video_model_type = gr.Radio(
                            choices=["faster-whisper", "whisperx", "openai"],
                            value="faster-whisper",
                            label="🤖 Model Type",
                            info="faster-whisper: fast & reliable | whisperx: 70x faster + diarization (requires: pip install whisperx) | openai: cloud API"
                        )
                        
                        video_model_size = gr.Dropdown(
                            choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
                            value="base",
                            label="📏 Model Size",
                            info="Larger = more accurate but slower"
                        )
                        
                        video_language = gr.Dropdown(
                            choices=["auto", "en", "es", "fr", "de", "it", "pt", "ru", "ja", "zh", "hi", "ar"],
                            value="en",
                            label="🌍 Language",
                            info="Select 'auto' for automatic detection"
                        )
                        
                        video_device = gr.Radio(
                            choices=["cpu", "cuda", "mps"],
                            value="cpu",
                            label="💻 Device",
                            info="cpu: All models | cuda: All models (NVIDIA GPU) | mps: WhisperX only (Apple M1/M2/M3)"
                        )
                        
                        video_enable_diarization = gr.Checkbox(
                            value=False,
                            label="👥 Enable Speaker Diarization (WhisperX only)",
                            info="Identifies different speakers (requires HF_TOKEN env variable)"
                        )
                        
                        # Speed multiplier removed as it's not relevant for batch processing
                        
                gr.Markdown("---")
                gr.Markdown("### 📹 Video Upload")
                
                gr.Markdown(
                    """
                    Enter the path to your video file to simulate a live meeting.  
                    The video will be processed as a batch job (fastest possible speed).
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
                
                start_sim_btn = gr.Button("▶️ Transcribe Video", variant="primary", size="lg")
                # stop_sim_btn removed as batch processing blocks until done (or we could implement cancellation)
                
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
                    placeholder="⏳ Transcript will appear here when processing is complete...\n\nUpload a video file and click 'Transcribe Video' to begin.",
                    lines=35,
                    max_lines=35,
                    interactive=False,
                    show_copy_button=True,
                    autoscroll=True
                )
                
                # Auto-update transcript while simulating
                def update_video_transcript():
                    while True:
                        if app.is_transcribing_video:
                            yield app.get_transcript()
                        time.sleep(0.5)
                
                # Timer to update video transcript
                interface.load(
                    fn=update_video_transcript,
                    outputs=[video_transcript_box],
                    every=1
                )
            
            # Tab 3: Analysis
            with gr.Tab("📊 Analysis"):
                gr.Markdown("### 🧠 AI Analysis")
                gr.Markdown("Analyze the transcript for sentiment and tone using BERT.")
                
                with gr.Row():
                    analyze_btn = gr.Button("🔍 Analyze Transcript", variant="primary", size="lg")
                
                with gr.Row():
                    with gr.Column():
                        analysis_status = gr.Textbox(label="Status", lines=2)
                    with gr.Column():
                        sentiment_label = gr.Textbox(label="Sentiment Label")
                        sentiment_score = gr.Number(label="Confidence Score")
                
                analyze_btn.click(
                    fn=app.analyze_transcript,
                    outputs=[analysis_status, sentiment_label, sentiment_score]
                )

            # Tab 4: Export & Settings
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
            inputs=[model_type, model_size, language, device, enable_diarization],
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
            fn=app.transcribe_video,
            inputs=[video_model_type, video_model_size, video_language, video_device, video_enable_diarization],
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
            
            ### 🎬 Video Transcription Mode
            
            1. **Upload Video**: Click "Upload Video File" and select a video (MP4, AVI, MOV, etc.)
            2. **Load Video**: Click "Load Video" to validate the file
            3. **Configure Settings**: Choose your transcription model and settings
            4. **Start Transcription**: Click "Transcribe Video" to begin
            5. **Wait for Result**: The system will process the entire video at once
            6. **Export**: Save the transcript from the Export & Settings tab
            
            **Note**: The video is processed as a batch job, which is much faster than real-time.
            
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
