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
