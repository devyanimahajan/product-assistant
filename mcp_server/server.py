"""
MCP Server for Product Discovery Assistant

Implements two tools:
1. rag.search - Query private Amazon Product Dataset 2020
2. web.search - Search web for current prices/availability (Brave Search API)

Following the MCP example structure with Flask and JSON-RPC 2.0
"""

import json
import os
from typing import Dict, Any
from flask import Flask, request, Response, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

from rag_tool import rag_search_tool
from web_tool import web_search_tool

load_dotenv()

app = Flask(__name__)
CORS(app)

MCP_TOOLS = [
    {
        "name": "rag.search",
        "description": "Query private Amazon Product Dataset 2020. Returns products with doc_id for citations. Use for historical product data, features, and baseline pricing.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language product search query"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results (default: 10)"
                },
                "filters": {
                    "type": "object",
                    "properties": {
                        "max_price": {
                            "type": "number",
                            "description": "Maximum price filter"
                        },
                        "min_rating": {
                            "type": "number",
                            "description": "Minimum rating filter"
                        },
                        "brand": {
                            "type": "string",
                            "description": "Filter by brand name"
                        },
                        "category": {
                            "type": "string",
                            "description": "Filter by category"
                        }
                    },
                    "description": "Optional metadata filters"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "web.search",
        "description": "Search web for current product prices and availability using Brave Search API. Returns live results with URLs for citations. Use when query mentions 'current', 'now', 'latest', 'availability'.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Product search query for live results"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum results (default: 5)"
                }
            },
            "required": ["query"]
        }
    }
]


def create_sse_message(data: Dict[str, Any]) -> str:
    """Format a message for Server-Sent Events (SSE)."""
    return f"data: {json.dumps(data)}\n\n"


def handle_initialize(message: Dict[str, Any]) -> Dict[str, Any]:
    """Handle MCP initialize request."""
    return {
        "jsonrpc": "2.0",
        "id": message.get("id"),
        "result": {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {},
            },
            "serverInfo": {
                "name": "product-discovery-mcp-server",
                "version": "1.0.0"
            }
        }
    }


def handle_tools_list(message: Dict[str, Any]) -> Dict[str, Any]:
    """Handle tools/list request."""
    return {
        "jsonrpc": "2.0",
        "id": message.get("id"),
        "result": {
            "tools": MCP_TOOLS
        }
    }


def handle_tools_call(message: Dict[str, Any]) -> Dict[str, Any]:
    """Handle tools/call request."""
    params = message.get("params", {})
    tool_name = params.get("name")
    arguments = params.get("arguments", {})
    
    if tool_name == "rag.search":
        result = rag_search_tool(**arguments)
    elif tool_name == "web.search":
        result = web_search_tool(**arguments)
    else:
        return {
            "jsonrpc": "2.0",
            "id": message.get("id"),
            "error": {
                "code": -32601,
                "message": f"Tool not found: {tool_name}"
            }
        }
    
    return {
        "jsonrpc": "2.0",
        "id": message.get("id"),
        "result": {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(result, indent=2)
                }
            ]
        }
    }


def process_mcp_message(message: Dict[str, Any]) -> Dict[str, Any]:
    """Process an MCP message and route to appropriate handler."""
    method = message.get("method")
    
    if method == "initialize":
        return handle_initialize(message)
    elif method == "tools/list":
        return handle_tools_list(message)
    elif method == "tools/call":
        return handle_tools_call(message)
    else:
        return {
            "jsonrpc": "2.0",
            "id": message.get("id"),
            "error": {
                "code": -32601,
                "message": f"Method not found: {method}"
            }
        }


# Flask Routes

@app.route('/mcp', methods=['POST'])
def mcp_endpoint():
    """Main MCP endpoint for communication."""
    message = request.get_json()
    
    def generate():
        try:
            response = process_mcp_message(message)
            yield create_sse_message(response)
        except Exception as e:
            error_response = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {
                    "code": -32700,
                    "message": f"Parse error: {str(e)}"
                }
            }
            yield create_sse_message(error_response)
    
    return Response(generate(), mimetype='text/event-stream')


@app.route('/tools/call', methods=['POST'])
def tools_call_endpoint():
    """Simple REST endpoint for tool calls (used by agents)."""
    data = request.get_json()
    tool_name = data.get("tool")
    arguments = data.get("arguments", {})
    
    print(f"\n[MCP] Tool call: {tool_name}")
    print(f"[MCP] Arguments: {arguments}")
    
    try:
        if tool_name == "rag.search":
            result = rag_search_tool(**arguments)
        elif tool_name == "web.search":
            print(f"[MCP] Calling web_search_tool...")
            result = web_search_tool(**arguments)
            print(f"[MCP] web_search_tool returned: success={result.get('success')}, count={result.get('count', 0)}")
        else:
            print(f"[MCP ERROR] Tool not found: {tool_name}")
            return jsonify({"error": f"Tool not found: {tool_name}"}), 404
        
        return jsonify({"result": result})
    except Exception as e:
        print(f"\n[MCP ERROR] Exception in tools_call_endpoint:")
        print(f"[MCP ERROR] Tool: {tool_name}")
        print(f"[MCP ERROR] Arguments: {arguments}")
        print(f"[MCP ERROR] Error: {type(e).__name__}: {e}")
        
        # Print full stack trace
        import traceback
        print(f"[MCP ERROR] Stack trace:")
        traceback.print_exc()
        
        return jsonify({"error": str(e)}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "server": "product-discovery-mcp-server",
        "version": "1.0.0",
        "tools": len(MCP_TOOLS)
    })


def start_mcp_server(host='127.0.0.1', port=8000):
    """Start the MCP server."""
    print(f"\n{'='*60}")
    print(f"Starting MCP Server")
    print(f"{'='*60}")
    print(f"Host: {host}:{port}")
    print(f"MCP Endpoint: http://{host}:{port}/mcp")
    print(f"Health Check: http://{host}:{port}/health")
    print(f"Available Tools: {len(MCP_TOOLS)}")
    for tool in MCP_TOOLS:
        print(f"  - {tool['name']}: {tool['description'][:60]}...")
    print(f"{'='*60}\n")
    
    app.run(host=host, port=port, debug=False, use_reloader=False)


if __name__ == "__main__":
    start_mcp_server()
