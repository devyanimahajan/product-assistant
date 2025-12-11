# Voice-to-Voice Product Assistant

## Setup (One-time)

```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys: OPENAI_API_KEY, BRAVE_API_KEY

# Build vector database
python data/build_chroma.py
```

## Run (Every time)

```bash
# Terminal 1: Start MCP Server
python mcp_server/server.py

# Terminal 2: Start Backend API (wait 5-6s for models to load)
python backend/app.py

# Terminal 3: Open Frontend
open frontend/index.html
```
