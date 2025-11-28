# WhisperX Setup Guide

WhisperX is an advanced transcription engine that provides:
- 🚀 70x realtime transcription speed
- 🎯 Word-level timestamps
- 👥 Speaker diarization (identifies who said what)
- 📊 Better accuracy through alignment

## Installation

### 1. Install WhisperX

```bash
# Activate your virtual environment first
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows

# Install WhisperX
pip install whisperx
```

### 2. Install CUDA (Optional, for GPU acceleration)

**For GPU support**, install CUDA 12.8:

- **Linux**: Follow [CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)
- **Windows**: Download from [CUDA Downloads](https://developer.nvidia.com/cuda-downloads)

### 3. Setup Speaker Diarization (Optional)

To enable speaker diarization (identify different speakers):

#### Step 1: Get HuggingFace Token
1. Go to [HuggingFace Settings](https://huggingface.co/settings/tokens)
2. Create a new token (read access is sufficient)
3. Copy the token

#### Step 2: Accept Model Agreements
Accept the user agreements for these models:
- [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
- [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)

#### Step 3: Set Environment Variable

**macOS/Linux:**
```bash
export HF_TOKEN='your-token-here'
```

**Windows (Command Prompt):**
```cmd
set HF_TOKEN=your-token-here
```

**Windows (PowerShell):**
```powershell
$env:HF_TOKEN='your-token-here'
```

**Permanent Setup (.bashrc or .zshrc):**
```bash
echo 'export HF_TOKEN="your-token-here"' >> ~/.bashrc
source ~/.bashrc
```

## Usage

### In the Web Interface

1. Start the application:
   ```bash
   python app.py
   ```

2. In the web UI:
   - Select **"whisperx"** as the Model Type
   - Choose your model size (base, small, medium, large-v2, large-v3)
   - Select language
   - Choose device (cpu or cuda)

3. For Speaker Diarization:
   - Ensure `HF_TOKEN` is set
   - The system will automatically use diarization when available
   - Transcripts will show speaker labels like `[SPEAKER_00] Hello there`

### Python API Example

```python
from src.live_transcription import LiveTranscriber

# Initialize with WhisperX
transcriber = LiveTranscriber(
    model_type="whisperx",
    model_size="base",
    language="en",
    device="cuda",  # or "cpu"
    compute_type="float16",
    enable_diarization=True,  # Requires HF_TOKEN
    hf_token="your-token-here"  # Or set HF_TOKEN environment variable
)

# Transcribe audio
result = transcriber.transcribe_chunk(audio_data, sample_rate=16000)
print(result)
```

## Features Comparison

| Feature | faster-whisper | WhisperX |
|---------|---------------|----------|
| Speed | Fast (5-10x realtime) | Very Fast (70x realtime) |
| Word Timestamps | Segment-level | Word-level |
| Speaker Diarization | ❌ | ✅ |
| Alignment Quality | Good | Excellent |
| Memory Usage | Lower | Slightly Higher |
| Setup Complexity | Simple | Moderate |

## Troubleshooting

### "whisperx not found"
```bash
pip install whisperx
```

### "Diarization failed"
- Ensure `HF_TOKEN` is set correctly
- Accept the pyannote model agreements
- Check internet connection (models download on first use)

### "CUDA out of memory"
- Use smaller model size (base or small)
- Reduce batch_size
- Use CPU instead: `device="cpu"`, `compute_type="int8"`

### Slow performance on CPU
- WhisperX is optimized for GPU
- Use `compute_type="int8"` for faster CPU inference
- Consider using faster-whisper for CPU-only setups

## Model Download Locations

Models are cached in:
- **Linux/macOS**: `~/.cache/huggingface/hub/`
- **Windows**: `C:\Users\{username}\.cache\huggingface\hub\`

## References

- [WhisperX GitHub](https://github.com/m-bain/whisperX)
- [WhisperX Paper](https://arxiv.org/abs/2303.00747)
- [PyAnnote Audio](https://github.com/pyannote/pyannote-audio)
