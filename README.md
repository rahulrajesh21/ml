# Live Meeting Transcription System

A real-time transcription system for meetings, lectures, and conversations using AI-powered speech recognition.

## 🌟 Features

- **Real-time Transcription**: Live audio transcription with minimal latency
- **Video Simulation**: Upload video files and simulate meeting transcription
- **Multiple Model Support**: 
  - Local transcription with faster-whisper (free, offline)
  - Cloud transcription with OpenAI Whisper API
- **Flexible Model Sizes**: Choose from tiny to large-v3 based on accuracy/speed needs
- **Multi-language Support**: Supports 12+ languages including English, Spanish, French, German, etc.
- **Export Options**: Download transcripts as TXT, JSON, or SRT subtitle format
- **User-friendly Interface**: Clean Gradio web interface with tabbed layout
- **Audio Device Selection**: List and select from available microphones
- **Timestamped Output**: Each transcription includes timestamp for easy reference
- **Speed Control**: Process videos faster than real-time with adjustable playback speed

## 📋 Requirements

- Python 3.8 or higher
- Microphone/audio input device (for live recording)
- **FFmpeg** (required for video simulation)
- **For macOS**: Homebrew (for PortAudio and FFmpeg installation)
- **For GPU acceleration**: NVIDIA GPU with CUDA support (optional)

## 🚀 Quick Start

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install system dependencies (required for audio/video processing)
# macOS
brew install portaudio ffmpeg

# Ubuntu/Debian
sudo apt-get install portaudio19-dev python3-pyaudio ffmpeg

# Install Python dependencies
pip install -r requirements.txt

# Run the application
python3 app.py
```

## 💻 Usage

1. **Open the web interface**: Navigate to `http://127.0.0.1:7860` in your browser

2. **Configure settings**:
   - **Model Type**: Choose `faster-whisper` (local) or `openai` (API)
   - **Model Size**: Select based on your needs:
     - `tiny/base`: Fast, real-time performance
     - `small/medium`: Better accuracy
     - `large-v3`: Best accuracy (requires GPU for real-time)
   - **Language**: Select your language or use `auto` for detection
   - **Device**: `cpu` or `cuda` (GPU)

3. **Check audio**: Click "List Audio Devices" to verify your microphone

4. **Start recording**: Click "▶️ Start Recording" and speak into your microphone

5. **Watch transcription**: Real-time text appears on the right (updates every 3 seconds)

6. **Stop recording**: Click "⏹️ Stop Recording" when finished

7. **Export**: Choose format (TXT/JSON/SRT) and click "📥 Download Transcript"

### Video Simulation

1. **Navigate to "Video Simulation" tab**

2. **Enter video file path**: Enter the full path to your video file (e.g., `/Users/you/video.mp4`)

3. **Process video**: Click "📹 Load Video" to validate the file

4. **Configure settings**: Select model, language, and playback speed
   - Speed multiplier: 1.0 = real-time, 2.0 = 2x faster processing

5. **Start simulation**: Click "▶️ Start Video Simulation"

6. **Watch transcription**: The video audio is processed in real-time chunks (3 seconds each)

7. **Export**: Use the Export tab to save the transcript

## 🎯 Model Selection Guide

| Model Size | Speed | Accuracy | Use Case | Memory |
|------------|-------|----------|----------|---------|
| tiny | ⚡⚡⚡⚡⚡ | ⭐⭐ | Quick notes, testing | ~1 GB |
| base | ⚡⚡⚡⚡ | ⭐⭐⭐ | Real-time meetings | ~1 GB |
| small | ⚡⚡⚡ | ⭐⭐⭐⭐ | Professional meetings | ~2 GB |
| medium | ⚡⚡ | ⭐⭐⭐⭐⭐ | Important recordings | ~5 GB |
| large-v3 | ⚡ | ⭐⭐⭐⭐⭐ | Critical transcription | ~10 GB |

## 📁 Project Structure

```
live-meeting-transcription/
├── app.py                    # Main Gradio web application
├── src/
│   ├── audio_capture.py     # Audio recording and buffering
│   └── live_transcription.py # Real-time transcription engine
├── exports/                  # Exported transcript files
├── requirements.txt          # Python dependencies
├── config.py                 # Configuration settings
└── README.md                 # This file
```

## ⚙️ Configuration

### Using OpenAI Whisper API

To use OpenAI's Whisper API:

1. Get an API key from [OpenAI Platform](https://platform.openai.com/)
2. Set the environment variable:
   ```bash
   export OPENAI_API_KEY='your-api-key-here'
   ```
3. Select "openai" as the model type in the web interface

### Using GPU Acceleration

For faster processing with GPU:

1. Ensure you have CUDA installed
2. Install GPU-compatible dependencies:
   ```bash
   pip install faster-whisper[gpu]
   ```
3. Select "cuda" as the device in the web interface

## 💡 Tips for Best Results

### Audio Quality
- Use a good quality microphone
- Minimize background noise
- Speak clearly at moderate pace
- Maintain consistent microphone distance
- Use headphones to prevent echo

### Performance
- Start with `base` model for testing
- Use GPU (CUDA) for larger models
- Each chunk is 3 seconds (configurable in code)
- Latency depends on model size and hardware

### Language Support
Supported languages include:
- English (en), Spanish (es), French (fr)
- German (de), Italian (it), Portuguese (pt)
- Russian (ru), Japanese (ja), Chinese (zh)
- Hindi (hi), Arabic (ar), and more

## 🔧 Troubleshooting

### "PyAudio not found" error
```bash
# macOS
brew install portaudio

# Ubuntu/Debian
sudo apt-get install portaudio19-dev
```

### No audio devices detected
- Check microphone permissions in System Preferences/Settings
- Ensure microphone is connected and recognized by OS
- Try running: `python3 -c "from src.audio_capture import AudioCapture; AudioCapture().list_audio_devices()"`

### Slow transcription
- Use smaller model (base/small)
- Enable GPU acceleration if available
- Reduce chunk duration in `audio_capture.py`

### Model download issues
- faster-whisper downloads models automatically on first use
- Models are cached in `~/.cache/huggingface/`
- Ensure stable internet connection for first run

### "FFmpeg not found" error
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

### Video simulation not working
- Ensure FFmpeg is installed and in PATH
- Verify video file has an audio track: `ffprobe your-video.mp4`
- Check file path is correct (use absolute path)
- Supported formats: MP4, AVI, MOV, MKV, WebM

## 📊 Export Formats

### TXT (Plain Text)
```
[14:30:15] Hello, this is a test transcription.

[14:30:18] The system is working well.
```

### JSON (Structured Data)
```json
{
  "export_timestamp": "20231126_143020",
  "entries": [
    {
      "timestamp": "14:30:15",
      "text": "Hello, this is a test transcription.",
      "chunk": 1
    }
  ],
  "word_count": 145
}
```

### SRT (Subtitle Format)
```
1
14:30:15,000 --> 14:30:15,000
Hello, this is a test transcription.

2
14:30:18,000 --> 14:30:18,000
The system is working well.
```

## 🛠️ Development

### Running Tests

```bash
# Test audio capture
python3 src/audio_capture.py

# Test transcription engine
python3 src/live_transcription.py
```

### Customization

Edit configuration in `app.py`:
- `chunk_duration`: Audio chunk size (default 3.0 seconds)
- `sample_rate`: Audio sample rate (default 16000 Hz)
- `server_port`: Web interface port (default 7860)

## 📝 License

This project is open source and available for educational and commercial use.

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest features
- Submit pull requests

## 🙏 Acknowledgments

- [faster-whisper](https://github.com/guillaumekln/faster-whisper) - Fast Whisper implementation
- [OpenAI Whisper](https://github.com/openai/whisper) - Speech recognition model
- [Gradio](https://gradio.app/) - Web interface framework
- [PyAudio](http://people.csail.mit.edu/hubert/pyaudio/) - Audio I/O library

## 📧 Support

For issues, questions, or suggestions, please open an issue on the project repository.

---

**Built with ❤️ for seamless meeting transcription**
