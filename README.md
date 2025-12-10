# product-assistant

## Setup
```bash
pip install -r requirements.txt
cp .env.example .env  # Add your BRAVE_API_KEY
python data/build_chroma.py
```
## Start MCP Server
```bash
python mcp_server/server.py
```
## ASR Setup 
1. Prerequisites
System Requirements:

- Python 3.8+
- FFmpeg (for audio conversion)

Install FFmpeg:
```bash
# Ubuntu/Debian
sudo apt update && sudo apt install ffmpeg

# macOS
brew install ffmpeg
```
2. Install Python Dependencies
```bash
# Clone repository (if not already done)
git clone <repository-url>
cd product-assistant

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt

# Or install ASR dependencies separately
pip install openai-whisper pydub numpy
```
3. Environment Configuration
Create .env file in project root:
```bash
# API Keys
BRAVE_API_KEY=your_brave_api_key_here
OPENAI_API_KEY=your_openai_key_here  # For future TTS/LLM

# ASR Configuration
WHISPER_MODEL_SIZE=base  # tiny/base/small/medium/large
WHISPER_DEVICE=cpu       # or 'cuda' for GPU

# MCP Server
MCP_HOST=127.0.0.1
MCP_PORT=5000
```
Get API keys:

- Brave Search API: https://brave.com/search/api/
- OpenAI API: https://platform.openai.com/

## Running ASR
A. Test ASR Component (Speech Recognition)
```bash
python asr_tool.py
```
This will:

Load Whisper model
Show supported languages
Wait for sample audio file

Test with your audio:
```bash
from asr_tool import transcribe_audio

result = transcribe_audio(
    audio_path="your_audio.wav",
    model_size="base",
    return_timestamps=True
)

print(f"Transcript: {result['text']}")
print(f"Language: {result['language']}")
```
## Quick Test ASR
```bash
# Option 1: Record and test
cd ASR
python record_audio.py test.wav 5

# Option 2: Test with existing audio
python quick_test_asr.py your_audio.wav
```

Expected output:
```
============================================================
ASR QUICK TEST
============================================================
✅ SUCCESS
📝 Transcript: "I need eco-friendly cleaner under fifteen dollars"
🌍 Language: EN
============================================================
```
