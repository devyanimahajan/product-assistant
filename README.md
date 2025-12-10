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