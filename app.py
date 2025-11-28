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
from src.text_analysis import TextAnalyzer, RoleBasedHighlightScorer
import threading
import queue


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

        # New attributes for the refactored approach
        self.highlight_scorer: Optional[RoleBasedHighlightScorer] = None
        self.is_recording = False
        self.audio_queue = queue.Queue()
        self.transcription_queue = queue.Queue()
        self.transcription_thread: Optional[threading.Thread] = None
        self.analysis_thread: Optional[threading.Thread] = None
        self.device: str = "cpu" # Default, will be set by UI
        self.model_type: str = "faster-whisper" # Default, will be set by UI
        self.model_size: str = "small" # Default, will be set by UI
        self.language: str = "en" # Default, will be set by UI
        self.enable_diarization: bool = False # Default, will be set by UI
        self.transcription_callback = None
        self.analysis_callback = None
        self.stop_event = threading.Event()

        # State for UI updates
        self.transcript_segments = [] # Stores dicts of {'text': ..., 'start': ..., 'end': ..., 'speaker': ...}
        self.highlights = []
        self.summary = ""
        self.action_items = ""
        self.sentiment_scores = {} # e.g., {'positive': 0.5, 'negative': 0.2, 'neutral': 0.3}

        # Speaker tracking for diarization
        self.current_speaker = "Speaker 1"
        self.speaker_change_count = 0
        self.speaker_timers = {} # {speaker_name: total_time_spoken}
        self.total_duration = 0.0
        self.last_segment_end_time = 0.0
        self.last_speaker_change_time = 0.0
        self.speaker_stats = {} # {speaker_name: {'time': ..., 'words': ...}}
        self.speaker_colors = ["#FFD700", "#87CEEB", "#90EE90", "#FF6347", "#BA55D3", "#FFA07A", "#20B2AA", "#DA70D6"]
        self.color_index = 0
        self.speaker_color_map = {} # {speaker_name: color_hex}
        self.speaker_id_counter = 1
        self.speaker_id_map = {} # {whisperx_speaker_label: "Speaker X"}
        self.speaker_id_reverse_map = {} # {"Speaker X": whisperx_speaker_label}
        self.speaker_id_colors = {} # {"Speaker X": color_hex}
        self.speaker_id_color_index = 0
        
        # Role Mapping
        self.role_mapping = {} # {"SPEAKER_01": "Product Manager"}
        self.role_embeddings = {} # {"Product Manager": [vector...]}

        
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
            # Initialize Text Analyzer
        self.text_analyzer = TextAnalyzer(device=self.device)
        
        # Initialize Highlight Scorer
        self.highlight_scorer = RoleBasedHighlightScorer(text_analyzer=self.text_analyzer)
        
        # State variables
        self.is_recording = False
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
    
    def get_role_vector(self, role: str) -> str:
        """
        Get the numerical vector for a role.
        
        Args:
            role: Role name
            
        Returns:
            Formatted string representation of the vector
        """
        # Lazy initialization of TextAnalyzer
        if not self.text_analyzer:
            print("Initializing Text Analyzer (Lazy Load)...")
            try:
                self.text_analyzer = TextAnalyzer(device=self.device)
            except Exception as e:
                return f"❌ Error initializing Text Analyzer: {str(e)}"
            
        embedding = self.text_analyzer.get_role_embedding(role)
        
        if not embedding:
            return "❌ Could not extract embedding."
            
        # Format for display
        dim = len(embedding)
        preview = ", ".join([f"{x:.4f}" for x in embedding[:8]])
        
        return f"Vector Dimension: {dim}\n\nFirst 8 values:\n[{preview}, ...]\n\nFull Vector (truncated):\n{str(embedding)[:500]}..."
    
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
                    # Apply mapping if available
                    display_name = self.get_mapped_speaker(speaker)
                    line = f"[{start_time}] [{display_name}] {text}\n\n"
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

    def analyze_transcript(self, manual_text=None) -> tuple[str, str, float]:
        """
        Analyze the current transcript using BERT.
        """
        # Use manual text if provided, otherwise use generated transcript
        text_to_analyze = manual_text if manual_text and manual_text.strip() else self.transcript_text
        
        if not text_to_analyze:
            return "⚠️ No transcript to analyze.", "N/A", 0.0
            
        # Lazy initialization of TextAnalyzer
        if not self.text_analyzer:
            print("Initializing Text Analyzer (Lazy Load)...")
            try:
                self.text_analyzer = TextAnalyzer(device=self.device)
            except Exception as e:
                return f"❌ Error initializing Text Analyzer: {str(e)}", "ERROR", 0.0
            
        try:
            print("Analyzing transcript sentiment...")
            result = self.text_analyzer.analyze_sentiment(text_to_analyze)
            
            label = result['label']
            score = result['score']
            
            status = f"✅ Analysis complete.\nSentiment: {label} ({score:.2f})"
            return status, label, score
            
        except Exception as e:
            return f"❌ Error analyzing transcript: {str(e)}", "ERROR", 0.0

    def generate_highlights(self, role: str, manual_text=None) -> str:
        """
        Generate highlights for a specific role.
        """
        text_to_analyze = manual_text if manual_text and manual_text.strip() else self.transcript_text
        
        if not text_to_analyze:
            return "⚠️ No transcript to analyze."
            
        # Lazy initialization of Highlight Scorer
        if not self.highlight_scorer:
            print("Initializing Highlight Scorer (Lazy Load)...")
            # Ensure TextAnalyzer is ready first
            if not self.text_analyzer:
                try:
                    self.text_analyzer = TextAnalyzer(device=self.device)
                except:
                    return "❌ Error initializing Text Analyzer for scoring."
            
            self.highlight_scorer = RoleBasedHighlightScorer(text_analyzer=self.text_analyzer)
            
        highlights = self.highlight_scorer.extract_highlights(text_to_analyze, role)
        
        if not highlights:
            return f"No specific highlights found for {role}."
            
        formatted_highlights = f"### ✨ Highlights for {role}\n\n"
        for i, highlight in enumerate(highlights, 1):
            formatted_highlights += f"**{i}.** {highlight}\n\n"
            
        return formatted_highlights





    def apply_role_mapping(self, json_str: str, current_transcript: str = "") -> tuple[str, str, str]:
        """
        Parse JSON mapping, generate embeddings, and update transcript with role names.
        """
        updated_transcript = current_transcript
        
        if not json_str or not json_str.strip():
            return "⚠️ Please enter a JSON mapping.", "", updated_transcript
            
        try:
            mapping = json.loads(json_str)
            if not isinstance(mapping, dict):
                return "❌ Invalid format. Expected a JSON object (dictionary).", "", updated_transcript
            
            self.role_mapping = mapping
            self.role_embeddings = {}
            
            # Lazy init analyzer
            if not self.text_analyzer:
                try:
                    self.text_analyzer = TextAnalyzer(device=self.device)
                except Exception as e:
                    return f"❌ Error initializing Text Analyzer: {str(e)}", "", updated_transcript
            
            # Generate embeddings
            embedding_info = "✅ Mapping Applied & Embeddings Generated:\n\n"
            
            for speaker_id, role in mapping.items():
                embedding = self.text_analyzer.get_embedding(role)
                if embedding is not None:
                    self.role_embeddings[role] = embedding
                    # Show first 5 dims
                    preview = ", ".join([f"{x:.4f}" for x in embedding[:5]])
                    embedding_info += f"🔹 {speaker_id} ➔ {role}\n"
                    embedding_info += f"   Vector (384-dim): [{preview}, ...]\n\n"
                else:
                    embedding_info += f"🔸 {speaker_id} ➔ {role} (Embedding failed)\n\n"
            
            # Update transcript text if available
            if updated_transcript:
                for speaker_id, role in mapping.items():
                    # Replace [SPEAKER_XX] with [Role]
                    # Simple string replace is safe enough for this format
                    updated_transcript = updated_transcript.replace(f"[{speaker_id}]", f"[{role}]")
                
                # Update internal state
                self.transcript_text = updated_transcript
            
            return "✅ Role mapping applied successfully!", embedding_info, updated_transcript
            
        except json.JSONDecodeError:
            return "❌ Invalid JSON format. Please check your syntax.", "", updated_transcript
        except Exception as e:
            return f"❌ Error applying mapping: {str(e)}", "", updated_transcript

    def get_mapped_speaker(self, speaker_label: str) -> str:
        """Get the mapped role name for a speaker label if it exists."""
        if speaker_label in self.role_mapping:
            return self.role_mapping[speaker_label]
        if speaker_label in self.role_mapping:
            return self.role_mapping[speaker_label]
        return speaker_label

    def generate_mapped_highlights(self, video_transcript: str, manual_transcript: str = "") -> str:
        """
        Generate highlights for all roles currently in the mapping.
        Prioritizes manual transcript if provided.
        """
        # Determine which text to use
        transcript_text = manual_transcript if manual_transcript and manual_transcript.strip() else video_transcript
        
        # Ensure mapping is applied to the text before analysis
        # This is critical for strict speaker filtering to work
        if self.role_mapping and transcript_text:
            for speaker_id, role in self.role_mapping.items():
                transcript_text = transcript_text.replace(f"[{speaker_id}]", f"[{role}]")
        
        if not self.role_mapping:
            return "⚠️ No roles mapped yet. Please apply a mapping first."
            
        if not transcript_text or not transcript_text.strip():
            return "⚠️ No transcript available to analyze. Please upload a video or paste a transcript."
            
        # Lazy init
        if not self.highlight_scorer:
            if not self.text_analyzer:
                try:
                    self.text_analyzer = TextAnalyzer(device=self.device)
                except Exception as e:
                    return f"❌ Error initializing analyzer: {str(e)}"
            self.highlight_scorer = RoleBasedHighlightScorer(text_analyzer=self.text_analyzer)
            
        # Get unique roles
        unique_roles = list(set(self.role_mapping.values()))
        
        full_output = ""
        
        for role in unique_roles:
            highlights = self.highlight_scorer.extract_highlights(transcript_text, role)
            if highlights:
                full_output += f"### ✨ Highlights for {role}\n\n"
                for i, h in enumerate(highlights, 1):
                    full_output += f"**{i}.** {h}\n\n"
            else:
                full_output += f"### ✨ Highlights for {role}\n*(No specific highlights found)*\n\n"
                
        return full_output

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

                        # Role Embedding Visualization (Safe Implementation)
                        with gr.Group():
                            gr.Markdown("### 🧠 View Role Embeddings (Phase 1)")
                            gr.Markdown("Visualize the numerical vector for a role.")
                            with gr.Row():
                                embed_role_input = gr.Textbox(label="Role Name", value="Developer")
                                embed_btn = gr.Button("🔢 Get Vector")
                            embed_output = gr.Textbox(label="Vector Output", lines=5)
                            
                            embed_btn.click(
                                fn=app.get_role_vector,
                                inputs=[embed_role_input],
                                outputs=[embed_output]
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
            
            # Tab 3: Role Mapping
            with gr.Tab("👥 Role Mapping"):
                gr.Markdown("### 🗺️ Map Speakers to Roles")
                gr.Markdown("Assign professional titles to speaker IDs (e.g., SPEAKER_01) and generate their semantic embeddings.")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        mapping_input = gr.Code(
                            language="json",
                            label="Speaker Mapping (JSON)",
                            value='{\n  "SPEAKER_00": "Developer",\n  "SPEAKER_01": "Product Manager",\n  "SPEAKER_02": "Designer"\n}',
                            lines=10
                        )
                        
                        apply_map_btn = gr.Button("🔄 Apply Mapping & Generate Embeddings", variant="primary")
                        
                    with gr.Column(scale=1):
                        mapping_status = gr.Textbox(label="Status", lines=2)
                        embedding_display = gr.Textbox(
                            label="Generated Embeddings", 
                            lines=15,
                            interactive=False,
                            info="Dense vector representations of the roles"
                        )
                
                apply_map_btn = gr.Button("🔄 Apply Mapping & Generate Embeddings", variant="primary")

                
                gr.Markdown("---")
                gr.Markdown("### ✨ Generate Role-Based Highlights")
                
                manual_transcript_input = gr.Textbox(
                    label="📝 Paste Transcript (Optional)",
                    placeholder="Paste your transcript here to analyze it directly (overrides video transcript)...",
                    lines=10
                )
                
                generate_mapped_btn = gr.Button("✨ Generate Highlights for Mapped Roles", size="lg")
                mapped_highlights_box = gr.Textbox(label="Mapped Highlights", lines=10)
                

            
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
                    manual_transcript_box = gr.Textbox(
                        label="📝 Transcript Input",
                        placeholder="Paste text here or load from generated transcript...",
                        lines=10
                    )
                
                with gr.Row():
                    load_transcript_btn = gr.Button("⬇️ Load Generated Transcript", size="sm")
                    analyze_btn = gr.Button("🔍 Analyze Transcript", variant="primary", size="lg")
                
                def refresh_manual_box():
                    return app.transcript_text
                    
                load_transcript_btn.click(
                    fn=refresh_manual_box,
                    outputs=[manual_transcript_box]
                )
                
                with gr.Row():
                    with gr.Column():
                        analysis_status = gr.Textbox(label="Status", lines=2)
                    with gr.Column():
                        sentiment_label = gr.Textbox(label="Sentiment Label")
                        sentiment_score = gr.Number(label="Confidence Score")
                
                analyze_btn.click(
                    fn=app.analyze_transcript,
                    inputs=[manual_transcript_box],
                    outputs=[analysis_status, sentiment_label, sentiment_score]
                )

                gr.Markdown("---")
                gr.Markdown("### ✨ Role-Based Highlights")
                
                with gr.Row():
                    role_dropdown = gr.Dropdown(
                        choices=["Developer", "Product Manager", "Designer", "QA Engineer", "Scrum Master"],
                        label="Select Role",
                        value="Developer"
                    )
                    highlight_btn = gr.Button("✨ Generate Highlights")
                
                highlights_box = gr.Textbox(label="Highlights", lines=5)
                
                highlight_btn.click(
                    fn=app.generate_highlights,
                    inputs=[role_dropdown, manual_transcript_box],
                    outputs=[highlights_box]
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
        
        # Event handlers - Mapped Highlights
        generate_mapped_btn.click(
            fn=app.generate_mapped_highlights,
            inputs=[video_transcript_box, manual_transcript_input],
            outputs=[mapped_highlights_box]
        )
        
        # Event handlers - Apply Mapping (Moved here to access video_transcript_box)
        apply_map_btn.click(
            fn=app.apply_role_mapping,
            inputs=[mapping_input, video_transcript_box],
            outputs=[mapping_status, embedding_display, video_transcript_box]
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
