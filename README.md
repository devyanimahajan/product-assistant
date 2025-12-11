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
# OpenAI
pip install openai>=1.0.0
```
Step 2. Configure API Keys

Copy and configure the environment file:

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```bash
# Choose provider
TTS_PROVIDER=openai 

# OpenAI (default)
OPENAI_API_KEY=sk-...
OPENAI_TTS_MODEL=tts-1  
OPENAI_TTS_VOICE=alloy
```
