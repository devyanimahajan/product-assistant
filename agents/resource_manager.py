"""
Resource Manager with Pre-loading and Singleton Pattern
Keeps expensive resources loaded in memory to minimize latency
"""

import sys
import os
from pathlib import Path
from typing import Optional

# Add paths for imports (agents dir must be first to avoid conflicts)
agents_dir = Path(__file__).parent
sys.path.insert(0, str(agents_dir.parent / 'ASR_Zoey'))
sys.path.insert(0, str(agents_dir.parent / 'TTS'))
sys.path.insert(0, str(agents_dir))  # Agents dir last (highest priority)

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / '.env')


class ResourceManager:
    """
    Singleton class that pre-loads and caches expensive resources:
    - ASR (Whisper model)
    - LLM client
    - MCP client
    - TTS engine
    - LangGraph workflow
    
    This eliminates initialization overhead on subsequent queries.
    """
    
    _instance: Optional['ResourceManager'] = None
    _initialized: bool = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ResourceManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize all resources once (singleton pattern)"""
        if ResourceManager._initialized:
            return
        
        print("[ResourceManager] Pre-loading all resources...")
        import time
        start = time.time()
        
        # Pre-load ASR
        print("  [1/5] Loading Whisper ASR model...")
        from asr_processor import ASRFragmentProcessor
        self.asr_processor = ASRFragmentProcessor()
        print(f"    ASR loaded on {self.asr_processor.device}")
        
        # Pre-load LLM
        print("  [2/5] Initializing LLM client...")
        from llm_factory import get_llm
        self.llm = get_llm()
        print("  - LLM client ready")
        
        # Pre-load MCP client
        print("  [3/5] Connecting to MCP server...")
        from mcp_client import MCPHTTPClient
        self.mcp_client = MCPHTTPClient()
        print("  - MCP client connected")
        
        # Pre-load TTS
        print("  [4/5] Initializing TTS engine...")
        from tts import TextToSpeech
        self.tts_engine = TextToSpeech(
            provider="openai",
            output_dir="test_audio",
            voice="nova"
        )
        print("  - TTS engine ready")
        
        # Pre-compile LangGraph
        print("  [5/5] Compiling LangGraph workflow...")
        from agent_state import build_assistant_graph
        self.graph = build_assistant_graph()
        print("  - LangGraph compiled")
        
        elapsed = time.time() - start
        print(f"[ResourceManager] All resources loaded in {elapsed:.2f}s")
        print(f"[ResourceManager] Subsequent queries will skip initialization\n")
        
        ResourceManager._initialized = True
    
    def get_asr(self):
        """Get pre-loaded ASR processor"""
        return self.asr_processor
    
    def get_llm(self):
        """Get pre-loaded LLM client"""
        return self.llm
    
    def get_mcp(self):
        """Get pre-loaded MCP client"""
        return self.mcp_client
    
    def get_tts(self):
        """Get pre-loaded TTS engine"""
        return self.tts_engine
    
    def get_graph(self):
        """Get pre-compiled LangGraph"""
        return self.graph
    
    @classmethod
    def reset(cls):
        """Reset singleton (useful for testing)"""
        cls._instance = None
        cls._initialized = False


def get_resource_manager() -> ResourceManager:
    """
    Get or create the singleton ResourceManager instance.
    First call will pre-load all resources (~5-7s).
    Subsequent calls return immediately.
    """
    return ResourceManager()
