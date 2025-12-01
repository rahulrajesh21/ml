
"""
Streamlit-based web interface for live meeting transcription.
Provides real-time transcription with controls and export functionality.
"""

import streamlit as st
import time
import threading
from typing import Optional, List, Dict
import os
import sys
import json
from datetime import datetime
import subprocess
import numpy as np
from dotenv import load_dotenv
import shutil

# Load environment variables
load_dotenv()

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.audio_capture import AudioCapture
from src.live_transcription import LiveTranscriber
from src.text_analysis import TextAnalyzer, RoleBasedHighlightScorer
from src.visual_analysis import VisualAnalyzer
from src.audio_analysis import AudioTonalAnalyzer, load_audio_file, LIBROSA_AVAILABLE
from src.fusion_layer import FusionLayer, SegmentFeatures
from src.video_processing import VideoSummarizer

# Page config
st.set_page_config(
    page_title="Live Meeting Transcription",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
def init_session_state():
    defaults = {
        'transcript_text': '',
        'video_audio_path': None,
        'is_transcribing': False,
        'text_analyzer': None,
        'audio_tonal_analyzer': None,
        'visual_analyzer': None,
        'fusion_layer': None,
        'cached_audio_data': None,
        'role_mapping': {},
        'role_embeddings': {},
        'transcriber': None,
        'device': 'cpu',
        'scored_segments': None # Store scored segments for video generation
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

def check_ffmpeg():
    """Check if FFmpeg is installed and accessible."""
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

if not check_ffmpeg():
    st.error("🚨 FFmpeg is not installed or not found in system PATH. Please install FFmpeg to use this app.")
    st.info("Windows: `winget install Gyan.FFmpeg` or download from ffmpeg.org and add to PATH.")
    st.stop()


# Helper functions
def get_text_analyzer():
    if st.session_state.text_analyzer is None:
        with st.spinner("Loading Text Analyzer..."):
            st.session_state.text_analyzer = TextAnalyzer(device=st.session_state.device)
    return st.session_state.text_analyzer

def get_audio_analyzer():
    if st.session_state.audio_tonal_analyzer is None and LIBROSA_AVAILABLE:
        with st.spinner("Loading Audio Analyzer..."):
            st.session_state.audio_tonal_analyzer = AudioTonalAnalyzer(sample_rate=16000)
    return st.session_state.audio_tonal_analyzer

def get_highlight_scorer():
    analyzer = get_text_analyzer()
    return RoleBasedHighlightScorer(text_analyzer=analyzer)

def parse_transcript_to_segments(transcript_text: str) -> List[Dict]:
    """Parse transcript text into segment dictionaries."""
    import re
    segments = []
    lines = transcript_text.strip().split('\n')
    pattern = re.compile(r'\[(\d{2}:\d{2}:\d{2})\]\s*(?:\[(.*?)\])?\s*(.*)')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        match = pattern.match(line)
        if match:
            timestamp_str = match.group(1)
            speaker = match.group(2)
            text = match.group(3).strip()
            if not text:
                continue
            parts = timestamp_str.split(':')
            seconds = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
            segments.append({
                'start': float(seconds),
                'end': float(seconds + 3),
                'text': text,
                'speaker': speaker
            })
    return segments

def transcribe_video(video_path, model_type, model_size, language, device, enable_diarization):
    """Transcribe video file."""
    openai_api_key = os.getenv("OPENAI_API_KEY") if model_type == "openai" else None
    hf_token = os.getenv("HF_TOKEN") if model_type == "whisperx" else None
    actual_model_size = "whisper-1" if model_type == "openai" else model_size
    
    transcriber = LiveTranscriber(
        model_type=model_type,
        model_size=actual_model_size,
        language=language if language != "auto" else None,
        openai_api_key=openai_api_key,
        hf_token=hf_token,
        device=device,
        enable_diarization=enable_diarization
    )
    st.session_state.transcriber = transcriber
    
    # Extract audio
    target_fps = 16000
    cmd = [
        'ffmpeg', '-i', video_path, '-vn', '-acodec', 'pcm_s16le',
        '-ar', str(target_fps), '-ac', '1', '-f', 's16le', '-loglevel', 'error', '-'
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    
    # Transcribe
    segments_generator = transcriber.transcribe_full_audio(result.stdout, target_fps)
    
    full_text = ""
    for segment in segments_generator:
        start_time = time.strftime('%H:%M:%S', time.gmtime(segment['start']))
        text = segment['text']
        speaker = segment.get('speaker')
        
        if speaker:
            mapped_name = st.session_state.role_mapping.get(speaker, speaker)
            line = f"[{start_time}] [{mapped_name}] {text}\n\n"
        else:
            line = f"[{start_time}] {text}\n\n"
        full_text += line
    
    return full_text


# Main UI
st.title("🎙️ Live Meeting Transcription System")
st.markdown("Real-time transcription with multi-modal fusion analysis")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    model_type = st.radio(
        "Model Type",
        ["faster-whisper", "whisperx", "openai"],
        help="faster-whisper: fast & reliable | whisperx: 70x faster + diarization | openai: cloud API"
    )
    
    model_size = st.selectbox(
        "Model Size",
        ["tiny", "base", "small", "medium", "large-v2", "large-v3"],
        index=1
    )
    
    language = st.selectbox(
        "Language",
        ["auto", "en", "es", "fr", "de", "it", "pt", "ru", "ja", "zh", "hi", "ar"],
        index=1
    )
    
    device = st.radio(
        "Device",
        ["cpu", "cuda", "mps"],
        help="cpu: All models | cuda: NVIDIA GPU | mps: Apple Silicon (WhisperX only)"
    )
    st.session_state.device = device
    
    enable_diarization = st.checkbox(
        "Enable Speaker Diarization",
        help="Identifies different speakers (WhisperX only, requires HF_TOKEN)"
    )
    
    st.divider()
    st.header("🔊 Audio Devices")
    if st.button("List Audio Devices"):
        try:
            temp_capture = AudioCapture()
            devices = temp_capture.list_audio_devices()
            temp_capture.cleanup()
            for d in devices:
                st.text(f"[{d['index']}] {d['name']}")
        except Exception as e:
            st.error(f"Error: {e}")

# Main tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🎙️ Live Recording",
    "🎬 Video Transcription",
    "👥 Role Mapping", 
    "📊 Analysis",
    "🔀 Fusion Analysis",
    "💾 Export"
])


# Tab 1: Live Recording
with tab1:
    st.header("🎙️ Live Recording")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("Record meeting audio directly.")
        
        if 'recording' not in st.session_state:
            st.session_state.recording = False
            
        if not st.session_state.recording:
            if st.button("🔴 Start Recording", type="primary"):
                st.session_state.recording = True
                st.session_state.audio_capture = AudioCapture()
                st.session_state.audio_capture.start_recording()
                st.rerun()
        else:
            st.error("Recording in progress...")
            if st.button("⏹️ Stop Recording"):
                st.session_state.recording = False
                if 'audio_capture' in st.session_state:
                    st.session_state.audio_capture.stop_recording()
                    # Save to file
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"recording_{timestamp}.wav"
                    os.makedirs("recordings", exist_ok=True)
                    filepath = os.path.join("recordings", filename)
                    st.session_state.audio_capture.save_wav(filepath)
                    st.session_state.audio_capture.cleanup()
                    del st.session_state.audio_capture
                    
                    st.session_state.video_audio_path = filepath
                    st.success(f"Recording saved: {filename}")
                    st.rerun()
    
    with col2:
        if st.session_state.recording:
            st.warning("🎙️ Microphone is active. Speak clearly.")
            st.spinner("Recording...")
        elif st.session_state.video_audio_path and st.session_state.video_audio_path.endswith(".wav"):
            st.audio(st.session_state.video_audio_path)
            
            if st.button("▶️ Transcribe Recording", type="primary"):
                 with st.spinner("Transcribing recording..."):
                    try:
                        # Reuse transcribe_video logic but for audio file
                        # We need to adapt transcribe_video slightly or just call transcriber directly
                        if st.session_state.transcriber is None:
                             st.session_state.transcriber = LiveTranscriber(
                                model_type=model_type,
                                model_size=model_size,
                                language=language if language != "auto" else None,
                                device=device,
                                enable_diarization=enable_diarization
                            )
                        
                        # Read audio file
                        with open(st.session_state.video_audio_path, "rb") as f:
                            audio_bytes = f.read()
                        
                        # Transcribe
                        # For simplicity, just use the full audio transcription
                        # We need to convert bytes to numpy for some methods, but let's use the file path if possible
                        # The existing transcribe_full_audio takes bytes or numpy
                        
                        import wave
                        with wave.open(st.session_state.video_audio_path, 'rb') as wf:
                            frames = wf.readframes(wf.getnframes())
                            sample_rate = wf.getframerate()
                            
                        segments_generator = st.session_state.transcriber.transcribe_full_audio(frames, sample_rate)
                        
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
                            
                        st.session_state.transcript_text = full_text
                        st.success("✅ Transcription complete!")
                        
                    except Exception as e:
                        st.error(f"Error: {e}")

# Tab 2: Video Transcription (Existing)
with tab2:
    st.header("🎬 Video Transcription")
    # ... (Existing video upload code) ...
    col1, col2 = st.columns([1, 2])
    
    with col1:
        uploaded_file = st.file_uploader("Upload Video", type=['mp4', 'mov', 'avi', 'mkv'])
        
        if uploaded_file is not None:
            # Save to temp directory
            os.makedirs("temp_uploads", exist_ok=True)
            file_path = os.path.join("temp_uploads", uploaded_file.name)
            
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.session_state.video_audio_path = file_path
            
            # Get duration
            try:
                from moviepy import VideoFileClip
                video = VideoFileClip(file_path)
                duration = video.duration
                video.close()
                st.success(f"✅ Video loaded! Duration: {duration:.1f}s")
            except Exception as e:
                st.warning(f"Could not read video metadata: {e}")

        st.divider()
        
        if st.button("▶️ Transcribe Video", type="primary", disabled=not st.session_state.video_audio_path):
            progress_bar = st.progress(0, text="Starting transcription...")
            
            with st.spinner("Transcribing video..."):
                try:
                    # Initialize transcriber
                    openai_api_key = os.getenv("OPENAI_API_KEY") if model_type == "openai" else None
                    hf_token = os.getenv("HF_TOKEN") if model_type == "whisperx" else None
                    actual_model_size = "whisper-1" if model_type == "openai" else model_size
                    
                    transcriber = LiveTranscriber(
                        model_type=model_type,
                        model_size=actual_model_size,
                        language=language if language != "auto" else None,
                        openai_api_key=openai_api_key,
                        hf_token=hf_token,
                        device=device,
                        enable_diarization=enable_diarization
                    )
                    st.session_state.transcriber = transcriber
                    
                    # Extract audio
                    progress_bar.progress(10, text="Extracting audio...")
                    target_fps = 16000
                    cmd = [
                        'ffmpeg', '-i', st.session_state.video_audio_path, '-vn', '-acodec', 'pcm_s16le',
                        '-ar', str(target_fps), '-ac', '1', '-f', 's16le', '-loglevel', 'error', '-'
                    ]
                    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
                    
                    # Transcribe
                    progress_bar.progress(20, text="Running AI model...")
                    segments_generator = transcriber.transcribe_full_audio(result.stdout, target_fps)
                    
                    full_text = ""
                    # We can't easily track progress of generator without consuming it
                    # But we can update periodically
                    
                    segments = list(segments_generator) # Consume generator
                    
                    for i, segment in enumerate(segments):
                        start_time = time.strftime('%H:%M:%S', time.gmtime(segment['start']))
                        text = segment['text']
                        speaker = segment.get('speaker')
                        
                        if speaker:
                            mapped_name = st.session_state.role_mapping.get(speaker, speaker)
                            line = f"[{start_time}] [{mapped_name}] {text}\n\n"
                        else:
                            line = f"[{start_time}] {text}\n\n"
                        full_text += line
                        
                        # Update progress
                        prog = 20 + int(70 * (i / len(segments)))
                        progress_bar.progress(min(prog, 90), text=f"Processing segment {i+1}...")
                    
                    st.session_state.transcript_text = full_text
                    
                    # Cache audio for fusion
                    if LIBROSA_AVAILABLE:
                        st.session_state.cached_audio_data = load_audio_file(
                            st.session_state.video_audio_path, sample_rate=16000
                        )
                    
                    progress_bar.progress(100, text="Done!")
                    st.success("✅ Transcription complete!")
                except Exception as e:
                    st.error(f"Error: {e}")
    
    with col2:
        st.subheader("📝 Transcript")
        transcript_display = st.text_area(
            "Transcript Output",
            value=st.session_state.transcript_text,
            height=500,
            label_visibility="collapsed"
        )
        if transcript_display != st.session_state.transcript_text:
            st.session_state.transcript_text = transcript_display

# Tab 3: Role Mapping (Existing)
with tab3:
    st.header("👥 Role Mapping")
    # ... (Existing role mapping code) ...
    st.markdown("Map speaker IDs to professional roles and generate semantic embeddings.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        default_mapping = '''{
  "SPEAKER_00": "Product Manager",
  "SPEAKER_01": "Developer",
  "SPEAKER_02": "Designer"
}'''
        mapping_json = st.text_area(
            "Speaker Mapping (JSON)",
            value=default_mapping,
            height=200
        )
        
        if st.button("🔄 Apply Mapping & Generate Embeddings", type="primary"):
            try:
                mapping = json.loads(mapping_json)
                st.session_state.role_mapping = mapping
                
                # Generate embeddings
                analyzer = get_text_analyzer()
                embeddings_info = []
                
                for speaker_id, role in mapping.items():
                    embedding = analyzer.get_embedding(role)
                    if embedding is not None:
                        st.session_state.role_embeddings[role] = embedding
                        preview = ", ".join([f"{x:.4f}" for x in embedding[:5]])
                        embeddings_info.append(f"✅ {speaker_id} → {role}\n   Vector: [{preview}, ...]")
                    else:
                        embeddings_info.append(f"❌ {speaker_id} → {role} (failed)")
                
                # Update transcript with role names
                if st.session_state.transcript_text:
                    updated = st.session_state.transcript_text
                    for speaker_id, role in mapping.items():
                        updated = updated.replace(f"[{speaker_id}]", f"[{role}]")
                    st.session_state.transcript_text = updated
                
                st.success("✅ Mapping applied!")
                st.session_state.embedding_info = "\n\n".join(embeddings_info)
                
            except json.JSONDecodeError:
                st.error("Invalid JSON format")
            except Exception as e:
                st.error(f"Error: {e}")
    
    with col2:
        st.subheader("Generated Embeddings")
        if 'embedding_info' in st.session_state:
            st.code(st.session_state.embedding_info)
        else:
            st.info("Apply mapping to see embeddings")

# Tab 4: Analysis (Updated with LLM)
with tab4:
    st.header("📊 Analysis")
    
    # LLM Summary Section
    st.subheader("🤖 AI Role-Based Summary")
    col1, col2 = st.columns([1, 3])
    with col1:
        summary_role = st.selectbox("Summary for Role", ["General", "Product Manager", "Developer", "Designer", "Executive"])
        summary_focus = st.text_input("Focus Area", value="key decisions and action items")
        
        if st.button("📝 Generate Summary"):
            if not st.session_state.transcript_text:
                st.warning("No transcript available.")
            else:
                with st.spinner("Generating AI summary..."):
                    try:
                        from src.llm_summarizer import LLMSummarizer
                        if 'llm_summarizer' not in st.session_state:
                            st.session_state.llm_summarizer = LLMSummarizer(device=st.session_state.device)
                        
                        summary = st.session_state.llm_summarizer.summarize(
                            st.session_state.transcript_text,
                            summary_role,
                            summary_focus
                        )
                        st.session_state.ai_summary = summary
                    except Exception as e:
                        st.error(f"Error: {e}")
    
    with col2:
        if 'ai_summary' in st.session_state:
            st.info(st.session_state.ai_summary)
        else:
            st.markdown("*Select a role and click Generate to see an AI-written summary.*")

    st.divider()
    
    # Sentiment Analysis (Existing)
    st.subheader("🧠 Sentiment Analysis")
    # ... (Existing sentiment code) ...
    manual_text = st.text_area(
        "Transcript to Analyze",
        value=st.session_state.transcript_text,
        height=200,
        key="sentiment_input"
    )
    
    if st.button("🔍 Analyze Sentiment"):
        if manual_text:
            with st.spinner("Analyzing..."):
                analyzer = get_text_analyzer()
                result = analyzer.analyze_sentiment(manual_text)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Sentiment", result['label'])
                with col2:
                    st.metric("Confidence", f"{result['score']:.2f}")
        else:
            st.warning("No text to analyze")
    
    st.divider()
    
    # Audio Tonal Analysis (Existing)
    st.subheader("🎵 Audio Tonal Analysis (MFCCs)")
    
    if st.button("🎵 Analyze Audio Tonal Features"):
        if not LIBROSA_AVAILABLE:
            st.error("librosa not installed. Run: pip install librosa")
        elif not st.session_state.video_audio_path:
            st.warning("Please load a video first")
        else:
            with st.spinner("Analyzing audio..."):
                try:
                    analyzer = get_audio_analyzer()
                    audio_data = load_audio_file(st.session_state.video_audio_path, sample_rate=16000)
                    
                    if audio_data is not None:
                        features = analyzer.extract_prosodic_features(audio_data)
                        emphasis_regions = analyzer.detect_emphasis_regions(audio_data)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Energy (Mean)", f"{features['energy_mean']:.4f}")
                            st.metric("Pitch (Mean)", f"{features['pitch_mean']:.1f} Hz")
                        with col2:
                            st.metric("Energy (Std)", f"{features['energy_std']:.4f}")
                            st.metric("Pitch (Std)", f"{features['pitch_std']:.1f} Hz")
                        with col3:
                            st.metric("Urgency Score", f"{features['urgency_score']:.2f}")
                            st.metric("Emphasis Score", f"{features['emphasis_score']:.2f}")
                        
                        if emphasis_regions:
                            st.markdown(f"**Emphasis Regions Detected:** {len(emphasis_regions)}")
                            for i, (start, end, intensity) in enumerate(emphasis_regions[:5], 1):
                                start_fmt = time.strftime('%H:%M:%S', time.gmtime(start))
                                end_fmt = time.strftime('%H:%M:%S', time.gmtime(end))
                                st.text(f"Region {i}: {start_fmt} - {end_fmt} (intensity: {intensity:.4f})")
                except Exception as e:
                    st.error(f"Error: {e}")

# Tab 5: Fusion Analysis (Updated with Dynamic Focus)
with tab5:
    st.header("🔀 Multi-Modal Fusion Analysis")
    st.markdown("Combine semantic (text), tonal (audio), and role signals for hyper-relevant highlights.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fusion_role = st.selectbox(
            "🎯 Target Role",
            ["Product Manager", "Developer", "Designer", "QA Engineer", "Scrum Master", "Tech Lead", "Data Scientist"],
            key="fusion_role"
        )
        
        fusion_strategy = st.radio(
            "⚙️ Fusion Strategy",
            ["weighted", "multiplicative", "gated"],
            help="weighted: linear combo | multiplicative: tonal boost | gated: role filters"
        )
        
        # NEW: Fusion Mode (Heuristic vs Neural)
        fusion_mode = st.radio(
            "🧠 Fusion Mode",
            ["Heuristic", "Neural (ML)"],
            help="Heuristic: Manual weights | Neural: Transformer model trained on importance"
        )
        
        # NEW: Dynamic Focus
        
        # NEW: Dynamic Focus
        focus_query = st.text_input("🔍 Custom Focus (Optional)", placeholder="e.g., budget concerns, deadlines")
    
    with col2:
        st.markdown("#### Fusion Weights")
        weight_semantic = st.slider("Semantic (What)", 0.0, 1.0, 0.5, 0.1)
        weight_tonal = st.slider("Tonal (How)", 0.0, 1.0, 0.2, 0.1)
        weight_role = st.slider("Role (Who)", 0.0, 1.0, 0.3, 0.1)
    
    if st.button("🔀 Run Fusion Analysis", type="primary"):
        if not st.session_state.transcript_text:
            st.warning("No transcript available. Please transcribe a video first.")
        else:
            with st.spinner("Running fusion analysis..."):
                try:
                    # Initialize components
                    text_analyzer = get_text_analyzer()
                    audio_analyzer = get_audio_analyzer() if LIBROSA_AVAILABLE else None
                    
                    # Normalize weights
                    total_weight = weight_semantic + weight_tonal + weight_role
                    if total_weight > 0:
                        weights = {
                            'semantic': weight_semantic / total_weight,
                            'tonal': weight_tonal / total_weight,
                            'role': weight_role / total_weight
                        }
                    else:
                        weights = {'semantic': 0.5, 'tonal': 0.2, 'role': 0.3}
                    
                    # Initialize fusion layer
                    fusion_layer = FusionLayer(
                        text_analyzer=text_analyzer,
                        audio_analyzer=audio_analyzer,
                        fusion_strategy=fusion_strategy,
                        weights=weights
                    )
                    
                    # Set role embeddings
                    if st.session_state.role_embeddings:
                        fusion_layer.set_role_embeddings(st.session_state.role_embeddings)
                    
                    # Parse segments
                    segments = parse_transcript_to_segments(st.session_state.transcript_text)
                    
                    if not segments:
                        st.warning("Could not parse transcript into segments")
                    else:
                        # Score segments with FOCUS QUERY
                        if fusion_mode == "Neural (ML)":
                            scored_segments = fusion_layer.score_segments_contextual(
                                segments,
                                fusion_role,
                                st.session_state.cached_audio_data,
                                sample_rate=16000,
                                focus_query=focus_query,
                                use_ml=True
                            )
                        else:
                            scored_segments = fusion_layer.score_segments(
                                segments,
                                fusion_role,
                                st.session_state.cached_audio_data,
                                sample_rate=16000,
                                focus_query=focus_query
                            )
                        
                        # Store for video generation
                        st.session_state.scored_segments = scored_segments
                        
                        # Get top segments
                        top_segments = fusion_layer.get_top_segments(scored_segments, top_n=10, min_score=0.2)
                        
                        # Display results
                        st.success(f"✅ Analyzed {len(segments)} segments")
                        
                        st.markdown(f"""
                        **Configuration:**
                        - Target Role: {fusion_role}
                        - Focus: {focus_query if focus_query else 'General Importance'}
                        - Strategy: {fusion_strategy}
                        """)
                        
                        st.divider()
                        st.subheader("🏆 Top Highlights")
                        
                        for i, seg in enumerate(top_segments, 1):
                            time_fmt = time.strftime('%H:%M:%S', time.gmtime(seg.start_time))
                            
                            with st.expander(f"**{i}. [{time_fmt}]** Score: {seg.fused_score:.2f}", expanded=(i <= 3)):
                                st.markdown(f"> {seg.text}")
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Semantic", f"{seg.semantic_score:.2f}")
                                with col2:
                                    st.metric("Tonal", f"{seg.tonal_score:.2f}")
                                with col3:
                                    st.metric("Role", f"{seg.role_relevance:.2f}")
                                
                                if seg.prosodic_features:
                                    st.caption(f"Urgency: {seg.prosodic_features.get('urgency_score', 0):.2f} | Emphasis: {seg.prosodic_features.get('emphasis_score', 0):.2f}")
                        
                        # Summary stats
                        if scored_segments:
                            st.divider()
                            st.subheader("📊 Summary Statistics")
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Avg Semantic", f"{np.mean([s.semantic_score for s in scored_segments]):.3f}")
                            with col2:
                                st.metric("Avg Tonal", f"{np.mean([s.tonal_score for s in scored_segments]):.3f}")
                            with col3:
                                st.metric("Avg Role", f"{np.mean([s.role_relevance for s in scored_segments]):.3f}")
                            with col4:
                                st.metric("Avg Fused", f"{np.mean([s.fused_score for s in scored_segments]):.3f}")
                
                except Exception as e:
                    import traceback
                    st.error(f"Error: {e}")
                    st.code(traceback.format_exc())

    # NEW: Training Section
    st.divider()
    st.subheader("🎓 Train Neural Model")
    st.markdown("Train the Neural Fusion model using the current transcript as a training example.")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        if st.button("🧠 Train Model (Self-Supervised/LLM)"):
            if not st.session_state.transcript_text:
                st.warning("No transcript available for training.")
            elif not st.session_state.scored_segments:
                st.warning("Run Fusion Analysis first to generate features.")
            else:
                with st.spinner("Training Neural Model..."):
                    try:
                        from src.train_fusion import FusionTrainer
                        from src.llm_summarizer import LLMSummarizer
                        
                        # Initialize LLM if needed
                        if 'llm_summarizer' not in st.session_state:
                            st.session_state.llm_summarizer = LLMSummarizer(device=st.session_state.device)
                            
                        trainer = FusionTrainer(st.session_state.llm_summarizer)
                        
                        # Train
                        loss = trainer.train_step(st.session_state.scored_segments, epochs=5)
                        
                        st.success(f"✅ Training Complete! Final Loss: {loss:.4f}")
                        st.info("Model saved to fusion_model.pth. Switch to 'Neural (ML)' mode to use it.")
                        
                        # Reload fusion layer to pick up new model
                        st.session_state.fusion_layer = None 
                        
                    except Exception as e:
                        import traceback
                        st.error(f"Training failed: {e}")
                        st.code(traceback.format_exc())

    st.divider()
    st.subheader("🎬 Generate Video Summary")
    
    if st.button("🎥 Generate Role-Specific Video Summary"):
        if not st.session_state.scored_segments:
            st.warning("Please run Fusion Analysis first to generate scores.")
        elif not st.session_state.video_audio_path:
            st.warning("No video file loaded.")
        else:
            with st.spinner("Generating smooth video summary..."):
                try:
                    scorer = get_highlight_scorer()
                    summarizer = VideoSummarizer(scorer)
                    
                    segments_for_smoothing = []
                    for seg in st.session_state.scored_segments:
                        segments_for_smoothing.append({
                            'start': seg.start_time,
                            'end': seg.end_time,
                            'score': seg.fused_score,
                            'text': seg.text
                        })
                    
                    # 1. Temporal Smoothing
                    time_ranges = summarizer.filter_and_smooth(
                        segments_for_smoothing,
                        threshold=0.4,
                        min_gap=2.0,
                        padding=1.0 # Increased padding for smoothness
                    )
                    
                    if not time_ranges:
                        st.warning("No segments met the threshold for the summary.")
                    else:
                        st.info(f"Generated {len(time_ranges)} clips after smoothing.")
                        
                        # 2. Generate Video with Crossfades
                        output_filename = f"summary_{fusion_role.replace(' ', '_')}_{int(time.time())}.mp4"
                        output_path = os.path.join("exports", output_filename)
                        os.makedirs("exports", exist_ok=True)
                        
                        result_path = summarizer.create_summary_video(
                            st.session_state.video_audio_path,
                            time_ranges,
                            output_path,
                            crossfade_duration=0.5 # Enable crossfade
                        )
                        
                        st.success(f"✅ Video Summary Generated: {output_filename}")
                        st.video(result_path)
                        
                        with open(result_path, "rb") as file:
                            st.download_button(
                                label="⬇️ Download Summary Video",
                                data=file,
                                file_name=output_filename,
                                mime="video/mp4"
                            )
                            
                except Exception as e:
                    import traceback
                    st.error(f"Error generating video: {e}")
                    st.code(traceback.format_exc())

# Tab 6: Export (Existing)
with tab6:
    st.header("💾 Export Transcript")
    # ... (Existing export code) ...
    export_format = st.radio(
        "Export Format",
        ["txt", "json", "srt"],
        horizontal=True,
        help="TXT: plain text | JSON: structured data | SRT: subtitle format"
    )
    
    if st.button("📥 Export Transcript", type="primary"):
        if not st.session_state.transcript_text:
            st.warning("No transcript to export")
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if export_format == "txt":
                content = f"LIVE MEETING TRANSCRIPT\n{'='*50}\n\n{st.session_state.transcript_text}"
                filename = f"transcript_{timestamp}.txt"
                mime = "text/plain"
                
            elif export_format == "json":
                segments = parse_transcript_to_segments(st.session_state.transcript_text)
                data = {
                    "export_timestamp": timestamp,
                    "segments": segments,
                    "full_text": st.session_state.transcript_text,
                    "word_count": len(st.session_state.transcript_text.split())
                }
                content = json.dumps(data, indent=2, ensure_ascii=False)
                filename = f"transcript_{timestamp}.json"
                mime = "application/json"
                
            elif export_format == "srt":
                segments = parse_transcript_to_segments(st.session_state.transcript_text)
                srt_lines = []
                for i, seg in enumerate(segments, 1):
                    start = time.strftime('%H:%M:%S', time.gmtime(seg['start']))
                    end = time.strftime('%H:%M:%S', time.gmtime(seg['end']))
                    srt_lines.append(f"{i}\n{start},000 --> {end},000\n{seg['text']}\n")
                content = "\n".join(srt_lines)
                filename = f"transcript_{timestamp}.srt"
                mime = "text/plain"
            
            st.download_button(
                label=f"⬇️ Download {filename}",
                data=content,
                file_name=filename,
                mime=mime
            )
            
            # Also save to exports folder
            os.makedirs("exports", exist_ok=True)
            filepath = os.path.join("exports", filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            st.success(f"✅ Saved to: {filepath}")

# Footer
st.divider()
st.markdown("""
---
### 📖 How to Use

1. **Load Video**: Upload a video file in the first tab.
2. **Transcribe**: Select model settings and click Transcribe Video.
3. **Map Roles**: Assign speaker IDs to professional roles.
4. **Analyze**: Run sentiment, highlights, or tonal analysis.
5. **Fusion**: Combine all signals for hyper-relevant highlights.
6. **Generate Video**: Create a temporally smoothed video summary for the target role.
7. **Export**: Download transcript in your preferred format.

**Tips:**
- Use `whisperx` with diarization for speaker identification
- Adjust fusion weights to prioritize different signals
- `gated` strategy strictly filters by role relevance
""")
