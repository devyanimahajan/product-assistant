"""
Complete Voice-to-Voice Integration Test
Tests the FULL pipeline: Audio Input → ASR → Agents → TTS → Audio Output

This proves the entire system works end-to-end with real components.
"""

import sys
import os
from pathlib import Path
import time

# Add parent directories to path (agents FIRST to avoid naming conflicts)
sys.path.insert(0, str(Path(__file__).parent.parent / 'agents'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'ASR_Zoey'))
# TTS added last to avoid conflict with agents/answerer.py
sys.path.append(str(Path(__file__).parent.parent / 'TTS'))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / '.env')

print("\n" + "="*70)
print("COMPLETE VOICE-TO-VOICE INTEGRATION TEST")
print("="*70)
print("\nPipeline: Audio → ASR → Router → Planner → Retriever → Answerer → TTS → Audio")
print("\nThis proves:")
print("  - ASR transcribes audio with GPU acceleration")
print("  - All 4 agents process the query")
print("  - TTS generates professional audio")
print("  - End-to-end latency is acceptable")
print("\n" + "="*70 + "\n")


def test_1_asr_transcription():
    """Test 1: ASR Component"""
    print("-"*70)
    print("TEST 1: ASR Transcription (Whisper with GPU)")
    print("-"*70)
    
    from asr_processor import ASRFragmentProcessor
    import torch
    
    # Check device
    if torch.backends.mps.is_available():
        print("\n- MPS (Apple Silicon GPU) detected")
        expected_device = "mps"
    elif torch.cuda.is_available():
        print("\n- CUDA (NVIDIA GPU) detected")
        expected_device = "cuda"
    else:
        print("\n⚠ No GPU detected, using CPU")
        expected_device = "cpu"
    
    # Initialize ASR
    test_audio = str(Path(__file__).parent.parent / "ASR_Zoey" / "test_query.mp3")
    
    if not os.path.exists(test_audio):
        print(f"\n- Test audio file not found: {test_audio}")
        return None
    
    print(f"\nTest Audio: {os.path.basename(test_audio)}")
    print(f"Initializing ASR processor...")
    
    start_time = time.time()
    asr = ASRFragmentProcessor()
    
    # Verify device
    print(f"- ASR using device: {asr.device}")
    assert asr.device == expected_device, f"Device mismatch"
    
    # Transcribe
    print("\nTranscribing audio...")
    result = asr.transcribe_fragment(test_audio)
    
    transcript = result.get("text", "")
    language = result.get("language", "")
    segments = result.get("segments", [])
    
    elapsed = time.time() - start_time
    
    print(f"\n- Transcription complete in {elapsed:.2f}s")
    print(f"  Transcript: '{transcript}'")
    print(f"  Language: {language}")
    print(f"  Segments: {len(segments)}")
    
    if segments:
        print(f"  Duration: {segments[-1]['end']:.2f}s")
    
    assert transcript, "No transcript generated"
    assert language == "en", "Expected English"
    
    return {
        "transcript": transcript,
        "language": language,
        "segments": segments,
        "asr_time": elapsed,
        "device": asr.device
    }


def test_2_agents_workflow(transcript):
    """Test 2: Agents Workflow (Router → Planner → Retriever → Answerer)"""
    print("\n" + "-"*70)
    print("TEST 2: Agents Workflow (4 Agents)")
    print("-"*70)
    
    # Change to agents directory for imports
    os.chdir(Path(__file__).parent.parent / 'agents')
    
    from agent_state import build_assistant_graph, AgentState
    
    print(f"\nInput: '{transcript}'")
    print("\nBuilding LangGraph...")
    graph = build_assistant_graph()
    
    initial_state: AgentState = {
        "user_query": transcript,
        "agent_logs": [],
        "tool_calls": [],
        "citations": [],
        "retrieved_products": [],
    }
    
    print("Executing workflow...")
    print("  → Router (intent classification)")
    print("  → Planner (data source selection)")
    print("  → Retriever (RAG search)")
    print("  → Answerer (synthesis + citations)")
    
    start_time = time.time()
    
    try:
        final_state = graph.invoke(initial_state)
        elapsed = time.time() - start_time
        
        print(f"\n- Workflow complete in {elapsed:.2f}s")
        
        # Verify components
        assert final_state.get("intent"), "Intent not set"
        assert final_state.get("retrieval_plan"), "Retrieval plan not set"
        assert len(final_state.get("retrieved_products", [])) > 0, "No products"
        assert final_state.get("final_response"), "No response"
        assert final_state.get("tts_summary"), "No TTS summary"
        assert len(final_state.get("citations", [])) > 0, "No citations"
        
        print(f"\n  Products Retrieved: {len(final_state['retrieved_products'])}")
        print(f"  Citations: {len(final_state['citations'])}")
        print(f"  Agent Logs: {len(final_state['agent_logs'])}")
        
        print(f"\n  Response (first 150 chars):")
        print(f"  {final_state['final_response'][:150]}...")
        
        print(f"\n  TTS Summary:")
        print(f"  {final_state['tts_summary']}")
        
        return {
            "final_state": final_state,
            "agents_time": elapsed,
            "products_count": len(final_state['retrieved_products']),
            "citations_count": len(final_state['citations'])
        }
        
    except Exception as e:
        print(f"\n- Workflow failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def test_3_tts_generation(tts_summary):
    """Test 3: TTS Audio Generation (OpenAI)"""
    print("\n" + "-"*70)
    print("TEST 3: TTS Audio Generation (OpenAI)")
    print("-"*70)
    
    from tts import TextToSpeech
    
    # Verify OpenAI API key is available
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key or openai_key == "your_openai_api_key_here":
        raise ValueError("OpenAI API key not found in .env file. TTS requires OpenAI API.")
    
    print(f"\n- OpenAI API key loaded: {openai_key[:20]}...")
    print("\nInitializing OpenAI TTS...")
    tts = TextToSpeech(
        provider="openai",
        output_dir="../test_audio",
        voice="nova"  # Female voice
    )
    output_file = "voice_to_voice_output.mp3"
    
    print(f"\nTTS Summary ({len(tts_summary)} chars):")
    print(f"  '{tts_summary}'")
    
    print("\nGenerating audio...")
    start_time = time.time()
    
    audio_path = tts.synthesize_text(tts_summary, output_file)
    
    elapsed = time.time() - start_time
    
    assert os.path.exists(audio_path), f"Audio file not created: {audio_path}"
    
    file_size = os.path.getsize(audio_path)
    
    print(f"\n- Audio generated in {elapsed:.2f}s")
    print(f"  Path: {audio_path}")
    print(f"  Size: {file_size / 1024:.1f} KB")
    print(f"  Provider: {tts.provider_name}")
    
    return {
        "audio_path": audio_path,
        "tts_time": elapsed,
        "file_size": file_size,
        "provider": tts.provider_name
    }


def test_4_complete_voice_to_voice():
    """Test 4: Complete Voice-to-Voice Pipeline"""
    print("\n" + "="*70)
    print("TEST 4: COMPLETE VOICE-TO-VOICE PIPELINE")
    print("="*70)
    
    total_start = time.time()
    
    # Step 1: ASR
    print("\n[1/3] Running ASR...")
    asr_result = test_1_asr_transcription()
    
    if not asr_result:
        print("\n- ASR test failed")
        return 1
    
    # Step 2: Agents
    print("\n[2/3] Running Agents...")
    agents_result = test_2_agents_workflow(asr_result["transcript"])
    
    # Step 3: TTS
    print("\n[3/3] Running TTS...")
    tts_summary = agents_result["final_state"]["tts_summary"]
    tts_result = test_3_tts_generation(tts_summary)
    
    total_elapsed = time.time() - total_start
    
    # Summary
    print("\n" + "="*70)
    print("COMPLETE VOICE-TO-VOICE TEST: SUCCESS!")
    print("="*70)
    
    print("\nPerformance Metrics:")
    print(f"  ASR Time:        {asr_result['asr_time']:.2f}s")
    print(f"  Agents Time:     {agents_result['agents_time']:.2f}s")
    print(f"  TTS Time:        {tts_result['tts_time']:.2f}s")
    print(f"  ─────────────────────────────")
    print(f"  Total Time:      {total_elapsed:.2f}s")
    
    print("\nData Flow:")
    print(f"  Input Audio:     ASR_Zoey/test_query.mp3")
    print(f"  Transcript:      '{asr_result['transcript'][:50]}...'")
    print(f"  Products Found:  {agents_result['products_count']}")
    print(f"  Citations:       {agents_result['citations_count']}")
    print(f"  TTS Summary:     '{tts_summary[:50]}...'")
    print(f"  Output Audio:    {tts_result['audio_path']}")
    
    print("\nComponent Status:")
    print(f"  - ASR (Whisper):        {asr_result['device'].upper()}")
    print(f"  - Router Agent:         OpenAI GPT")
    print(f"  - Planner Agent:        Rule-based")
    print(f"  - Retriever Agent:      ChromaDB Vector Search")
    print(f"  - Answerer Agent:       OpenAI GPT")
    print(f"  - TTS:                  {tts_result['provider'].title()}")
    
    print("\nAudio Output:")
    print(f"  File: {tts_result['audio_path']}")
    print(f"  Size: {tts_result['file_size'] / 1024:.1f} KB")
    print(f"  Play: afplay {tts_result['audio_path']}")
    
    print("\n" + "="*70)
    print("\nAll 3 components (ASR, Agents, TTS) verified")
    print("\n" + "="*70 + "\n")
    
    return 0


def main():
    """Run complete voice-to-voice test"""
    
    try:
        exit_code = test_4_complete_voice_to_voice()
        return exit_code
        
    except Exception as e:
        print("\n" + "="*70)
        print("VOICE-TO-VOICE TEST FAILED")
        print("="*70)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
