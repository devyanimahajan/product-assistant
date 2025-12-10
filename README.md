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

## TTS Setup
Step 1: Install Dependencies
```bash
pip install transformers datasets soundfile torch torchaudio
```
Step 2: Verify Installation
```bash
python -c "import transformers; import datasets; import soundfile; import torch; print('✓ All packages installed')"
```

## Run TTS
```bash
# Test TTS
python tts/test_local_tts.py

# Or test directly
python -c "from tts.local_tts import LocalTTS; tts = LocalTTS(); print('✓ TTS ready')"
```
Files

tts/local_tts.py - TTS module

agents/answerer.py - Integration with answerer agent
