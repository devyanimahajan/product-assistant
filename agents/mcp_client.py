import os
from typing import Any, Dict
import requests


class MCPHTTPClient:
    """
    Minimal HTTP client for MCP tools over a simple JSON HTTP API.

    Expected server:
      POST {base_url}/tools/call
      body: {"tool": "rag.search", "arguments": {...}}
      response: {"result": {...}} or direct tool result.
    """

    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = (base_url or os.getenv("MCP_SERVER_URL", "http://localhost:8000")).rstrip("/")

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        payload = {
            "tool": tool_name,
            "arguments": arguments,
        }
        resp = requests.post(f"{self.base_url}/tools/call", json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict) and "result" in data:
            return data["result"]
        return data
