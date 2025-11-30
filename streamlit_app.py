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
        'device': 'cpu'
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()


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
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎬 Video Transcription",
    "👥 Role Mapping", 
    "📊 Analysis",
    "🔀 Fusion Analysis",
    "💾 Export"
])


# Tab 1: Video Transcription
with tab1:
    st.header("🎬 Video Transcription")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        video_path = st.text_input(
            "Video File Path",
            placeholder="/path/to/your/video.mp4",
            help="Enter the full path to your video file"
        )
        
        if st.button("📂 Load Video", type="secondary"):
            if video_path and os.path.exists(video_path):
                try:
                    from moviepy import VideoFileClip
                    video = VideoFileClip(video_path)
                    if video.audio is None:
                        st.error("Video has no audio track")
                    else:
                        duration = video.duration
                        st.session_state.video_audio_path = video_path
                        st.success(f"✅ Video loaded! Duration: {duration:.1f}s")
                    video.close()
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.warning("Please enter a valid video path")
        
        st.divider()
        
        if st.button("▶️ Transcribe Video", type="primary", disabled=not st.session_state.video_audio_path):
            with st.spinner("Transcribing video... This may take a while."):
                try:
                    transcript = transcribe_video(
                        st.session_state.video_audio_path,
                        model_type, model_size, language, device, enable_diarization
                    )
                    st.session_state.transcript_text = transcript
                    
                    # Cache audio for fusion
                    if LIBROSA_AVAILABLE:
                        st.session_state.cached_audio_data = load_audio_file(
                            st.session_state.video_audio_path, sample_rate=16000
                        )
                    
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


# Tab 2: Role Mapping
with tab2:
    st.header("👥 Role Mapping")
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


# Tab 3: Analysis
with tab3:
    st.header("📊 Analysis")
    
    # Sentiment Analysis
    st.subheader("🧠 Sentiment Analysis")
    
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
    
    # Role-Based Highlights
    st.subheader("✨ Role-Based Highlights")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        role_select = st.selectbox(
            "Select Role",
            ["Developer", "Product Manager", "Designer", "QA Engineer", "Scrum Master", "Tech Lead"]
        )
        
        if st.button("✨ Generate Highlights"):
            if st.session_state.transcript_text:
                with st.spinner("Generating highlights..."):
                    scorer = get_highlight_scorer()
                    highlights = scorer.extract_highlights(st.session_state.transcript_text, role_select)
                    st.session_state.highlights = highlights
            else:
                st.warning("No transcript available")
    
    with col2:
        if 'highlights' in st.session_state and st.session_state.highlights:
            for i, (text, score) in enumerate(st.session_state.highlights, 1):
                st.markdown(f"**{i}.** (Score: {score:.2f}) {text}")
        else:
            st.info("Generate highlights to see results")
    
    st.divider()
    
    # Audio Tonal Analysis
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


# Tab 4: Fusion Analysis
with tab4:
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
                        # Score segments
                        scored_segments = fusion_layer.score_segments(
                            segments,
                            fusion_role,
                            st.session_state.cached_audio_data,
                            sample_rate=16000
                        )
                        
                        # Get top segments
                        top_segments = fusion_layer.get_top_segments(scored_segments, top_n=10, min_score=0.2)
                        
                        # Display results
                        st.success(f"✅ Analyzed {len(segments)} segments")
                        
                        st.markdown(f"""
                        **Configuration:**
                        - Target Role: {fusion_role}
                        - Strategy: {fusion_strategy}
                        - Weights: Semantic={weights['semantic']:.2f}, Tonal={weights['tonal']:.2f}, Role={weights['role']:.2f}
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


# Tab 5: Export
with tab5:
    st.header("💾 Export Transcript")
    
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

1. **Load Video**: Enter path and click Load Video
2. **Transcribe**: Select model settings and click Transcribe Video
3. **Map Roles**: Assign speaker IDs to professional roles
4. **Analyze**: Run sentiment, highlights, or tonal analysis
5. **Fusion**: Combine all signals for hyper-relevant highlights
6. **Export**: Download transcript in your preferred format

**Tips:**
- Use `whisperx` with diarization for speaker identification
- Adjust fusion weights to prioritize different signals
- `gated` strategy strictly filters by role relevance
""")
