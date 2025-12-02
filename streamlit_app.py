"""
RoME: Role-aware Multimodal Meeting Summarizer
Professional Interface
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
    page_title="RoME System",
    page_icon=None,
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
        'scored_segments': None,
        'ai_summary': None
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
    st.error("FFmpeg is not installed or not found in system PATH.")
    st.stop()

# Helper functions
def get_text_analyzer():
    if st.session_state.text_analyzer is None:
        with st.spinner("Initializing Text Analyzer..."):
            st.session_state.text_analyzer = TextAnalyzer(device=st.session_state.device)
    return st.session_state.text_analyzer

def get_audio_analyzer():
    if st.session_state.audio_tonal_analyzer is None and LIBROSA_AVAILABLE:
        with st.spinner("Initializing Audio Analyzer..."):
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

# --- Sidebar Configuration ---
with st.sidebar:
    st.title("RoME Configuration")
    
    st.subheader("Model Settings")
    model_type = st.selectbox(
        "Transcription Model",
        ["faster-whisper", "whisperx", "openai"],
        help="Select the backend for speech-to-text."
    )
    
    model_size = st.selectbox(
        "Model Size",
        ["tiny", "base", "small", "medium", "large-v2", "large-v3"],
        index=1
    )
    
    device = st.selectbox(
        "Compute Device",
        ["cpu", "cuda", "mps"],
        index=0
    )
    st.session_state.device = device
    
    enable_diarization = st.checkbox(
        "Enable Speaker Diarization",
        value=True,
        help="Identify unique speakers in the audio."
    )
    
    st.divider()
    st.info("System Ready")

# --- Main Interface ---
st.title("RoME: Role-aware Multimodal Meeting Summarizer")
st.markdown("### End-to-End Pipeline")

# Section 1: Input & Preprocessing
st.header("1. Input & Preprocessing")
st.markdown("Upload video, extract audio, transcribe, and map speakers.")

col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader("Upload Meeting Video", type=['mp4', 'mov', 'avi', 'mkv'])
    
    if uploaded_file is not None:
        os.makedirs("temp_uploads", exist_ok=True)
        file_path = os.path.join("temp_uploads", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.session_state.video_audio_path = file_path
        st.success(f"Video loaded: {uploaded_file.name}")

with col2:
    if st.button("Process Video (Transcribe & Diarize)", disabled=not st.session_state.video_audio_path):
        progress_bar = st.progress(0, text="Initializing...")
        
        try:
            # 1. Initialize Transcriber
            openai_api_key = os.getenv("OPENAI_API_KEY") if model_type == "openai" else None
            hf_token = os.getenv("HF_TOKEN") if model_type == "whisperx" else None
            actual_model_size = "whisper-1" if model_type == "openai" else model_size
            
            transcriber = LiveTranscriber(
                model_type=model_type,
                model_size=actual_model_size,
                openai_api_key=openai_api_key,
                hf_token=hf_token,
                device=device,
                enable_diarization=enable_diarization
            )
            st.session_state.transcriber = transcriber
            
            # 2. Extract Audio
            progress_bar.progress(20, text="Extracting Audio Stream...")
            target_fps = 16000
            cmd = [
                'ffmpeg', '-i', st.session_state.video_audio_path, '-vn', '-acodec', 'pcm_s16le',
                '-ar', str(target_fps), '-ac', '1', '-f', 's16le', '-loglevel', 'error', '-'
            ]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            
            # 3. Transcribe & Diarize
            progress_bar.progress(40, text="Transcribing and Diarizing...")
            segments_generator = transcriber.transcribe_full_audio(result.stdout, target_fps)
            
            full_text = ""
            segments = list(segments_generator)
            
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
                
                prog = 40 + int(50 * (i / len(segments)))
                progress_bar.progress(min(prog, 90), text=f"Processing segment {i+1}...")
            
            st.session_state.transcript_text = full_text
            
            # 4. Cache Audio for Fusion
            if LIBROSA_AVAILABLE:
                st.session_state.cached_audio_data = load_audio_file(
                    st.session_state.video_audio_path, sample_rate=16000
                )
            
            progress_bar.progress(100, text="Complete")
            st.success("Preprocessing Complete")
            
        except Exception as e:
            st.error(f"Processing Error: {e}")

# Role Mapping
if st.session_state.transcript_text:
    with st.expander("Speaker Role Mapping", expanded=True):
        col_map1, col_map2 = st.columns(2)
        with col_map1:
            default_mapping = '{\n  "SPEAKER_00": "Product Manager",\n  "SPEAKER_01": "Developer",\n  "SPEAKER_02": "Designer"\n}'
            mapping_json = st.text_area("Role Map (JSON)", value=default_mapping, height=150)
            if st.button("Apply Role Mapping"):
                try:
                    mapping = json.loads(mapping_json)
                    st.session_state.role_mapping = mapping
                    
                    # Generate Embeddings
                    analyzer = get_text_analyzer()
                    for speaker_id, role in mapping.items():
                        emb = analyzer.get_embedding(role)
                        if emb is not None:
                            st.session_state.role_embeddings[role] = emb
                    
                    # Update Transcript
                    updated = st.session_state.transcript_text
                    for speaker_id, role in mapping.items():
                        updated = updated.replace(f"[{speaker_id}]", f"[{role}]")
                    st.session_state.transcript_text = updated
                    st.success("Roles Mapped and Embeddings Generated")
                except Exception as e:
                    st.error(f"Mapping Error: {e}")
        
        with col_map2:
            st.text_area("Current Transcript", value=st.session_state.transcript_text, height=200, disabled=True)

st.divider()

# Section 2: Core Processing
st.header("2. Core Processing (Fusion)")
st.markdown("Configure multimodal fusion to identify role-relevant highlights.")

col_core1, col_core2 = st.columns(2)

with col_core1:
    target_role = st.selectbox(
        "Target Audience Role",
        ["Product Manager", "Developer", "Designer", "QA Engineer", "Executive"],
        index=0
    )
    
    fusion_mode = st.radio(
        "Fusion Mode",
        ["Heuristic (Manual Weights)", "Neural (Transformer Model)"],
        horizontal=True
    )
    
    focus_query = st.text_input("Custom Focus Area (Optional)", placeholder="e.g., budget, deadlines, technical debt")

with col_core2:
    st.markdown("**Fusion Weights (Heuristic Mode)**")
    w_semantic = st.slider("Semantic (Text)", 0.0, 1.0, 0.5)
    w_tonal = st.slider("Tonal (Audio)", 0.0, 1.0, 0.2)
    w_role = st.slider("Role (Context)", 0.0, 1.0, 0.3)

if st.button("Run Multimodal Analysis", type="primary"):
    if not st.session_state.transcript_text:
        st.warning("Please complete Step 1 first.")
    else:
        with st.spinner("Running Fusion Analysis..."):
            try:
                text_analyzer = get_text_analyzer()
                audio_analyzer = get_audio_analyzer()
                
                # Weights
                total = w_semantic + w_tonal + w_role
                weights = {
                    'semantic': w_semantic/total if total > 0 else 0.5,
                    'tonal': w_tonal/total if total > 0 else 0.2,
                    'role': w_role/total if total > 0 else 0.3
                }
                
                fusion_layer = FusionLayer(
                    text_analyzer=text_analyzer,
                    audio_analyzer=audio_analyzer,
                    weights=weights
                )
                if st.session_state.role_embeddings:
                    fusion_layer.set_role_embeddings(st.session_state.role_embeddings)
                
                segments = parse_transcript_to_segments(st.session_state.transcript_text)
                
                if fusion_mode.startswith("Neural"):
                     scored_segments = fusion_layer.score_segments_contextual(
                        segments, target_role, st.session_state.cached_audio_data,
                        sample_rate=16000, focus_query=focus_query, use_ml=True
                    )
                else:
                    scored_segments = fusion_layer.score_segments(
                        segments, target_role, st.session_state.cached_audio_data,
                        sample_rate=16000, focus_query=focus_query
                    )
                
                st.session_state.scored_segments = scored_segments
                st.success(f"Analysis Complete. Processed {len(segments)} segments.")
                
            except Exception as e:
                st.error(f"Analysis Failed: {e}")

# Advanced: Training
with st.expander("Advanced: Model Training"):
    st.markdown("Train the Neural Fusion model on the current transcript.")
    if st.button("Train Model"):
        if st.session_state.scored_segments:
            with st.spinner("Training..."):
                try:
                    from src.train_fusion import FusionTrainer
                    from src.llm_summarizer import LLMSummarizer
                    
                    if 'llm_summarizer' not in st.session_state:
                        st.session_state.llm_summarizer = LLMSummarizer(device=st.session_state.device)
                    
                    trainer = FusionTrainer(st.session_state.llm_summarizer)
                    loss = trainer.train_step(st.session_state.scored_segments, epochs=5)
                    st.success(f"Training Complete. Loss: {loss:.4f}")
                except Exception as e:
                    st.error(f"Training Error: {e}")
        else:
            st.warning("Run Analysis first.")

st.divider()

# Section 3: Generation
st.header("3. Generation & Results")
st.markdown("Generate role-specific video highlights and text summaries.")

col_gen1, col_gen2 = st.columns(2)

with col_gen1:
    st.subheader("Video Highlight")
    if st.button("Generate Video Digest"):
        if not st.session_state.scored_segments:
            st.warning("Run Analysis first.")
        elif not st.session_state.video_audio_path:
            st.warning("No video loaded.")
        else:
            with st.spinner("Generating Video..."):
                try:
                    scorer = get_highlight_scorer()
                    summarizer = VideoSummarizer(scorer)
                    
                    # Prepare segments
                    seg_data = [{
                        'start': s.start_time, 'end': s.end_time,
                        'score': s.fused_score, 'text': s.text
                    } for s in st.session_state.scored_segments]
                    
                    # Smooth
                    ranges = summarizer.filter_and_smooth(seg_data, threshold=0.4, min_gap=2.0, padding=1.0)
                    
                    if ranges:
                        output_filename = f"summary_{target_role.replace(' ', '_')}_{int(time.time())}.mp4"
                        output_path = os.path.join("exports", output_filename)
                        os.makedirs("exports", exist_ok=True)
                        
                        result_path = summarizer.create_summary_video(
                            st.session_state.video_audio_path, ranges, output_path, crossfade_duration=0.5
                        )
                        st.video(result_path)
                        st.success(f"Video Generated: {output_filename}")
                    else:
                        st.warning("No significant highlights found.")
                except Exception as e:
                    st.error(f"Generation Error: {e}")

with col_gen2:
    st.subheader("Text Summary")
    if st.button("Generate Text Summary"):
        if not st.session_state.transcript_text:
            st.warning("No transcript.")
        else:
            with st.spinner("Generating Summary..."):
                try:
                    from src.llm_summarizer import LLMSummarizer
                    if 'llm_summarizer' not in st.session_state:
                        st.session_state.llm_summarizer = LLMSummarizer(device=st.session_state.device)
                    
                    summary = st.session_state.llm_summarizer.summarize(
                        st.session_state.transcript_text, target_role, focus_query or "key points"
                    )
                    st.session_state.ai_summary = summary
                except Exception as e:
                    st.error(f"Summary Error: {e}")
    
    if st.session_state.ai_summary:
        st.info(st.session_state.ai_summary)
