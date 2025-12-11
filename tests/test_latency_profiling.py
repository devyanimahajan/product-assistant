"""
Detailed Latency Profiling Test
Measures time spent in each component to identify bottlenecks
"""

import sys
import os
from pathlib import Path
import time

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'agents'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'ASR_Zoey'))
sys.path.append(str(Path(__file__).parent.parent / 'TTS'))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / '.env')

print("\n" + "="*70)
print("LATENCY PROFILING TEST")
print("="*70)
print("\nMeasuring individual component latency...")
print("\n" + "="*70 + "\n")


def profile_asr():
    """Profile ASR component"""
    print("-"*70)
    print("PROFILING: ASR (Whisper)")
    print("-"*70)
    
    from asr_processor import ASRFragmentProcessor
    
    test_audio = str(Path(__file__).parent.parent / "ASR_Zoey" / "test_query.mp3")
    
    # Measure model loading
    print("\n[1] Loading Whisper model...")
    start = time.time()
    asr = ASRFragmentProcessor()
    load_time = time.time() - start
    print(f"✓ Model loaded in {load_time:.2f}s")
    
    # Measure transcription
    print("\n[2] Transcribing audio...")
    start = time.time()
    result = asr.transcribe_fragment(test_audio)
    transcribe_time = time.time() - start
    print(f"✓ Transcription in {transcribe_time:.2f}s")
    
    total = load_time + transcribe_time
    print(f"\n  Model Load:     {load_time:.2f}s ({load_time/total*100:.1f}%)")
    print(f"  Transcription:  {transcribe_time:.2f}s ({transcribe_time/total*100:.1f}%)")
    print(f"  ─────────────────────────────")
    print(f"  Total ASR:      {total:.2f}s")
    
    return {
        "load_time": load_time,
        "transcribe_time": transcribe_time,
        "total": total,
        "transcript": result.get("text", "")
    }


def profile_agents(transcript):
    """Profile each agent individually"""
    print("\n" + "-"*70)
    print("PROFILING: Agent Workflow")
    print("-"*70)
    
    os.chdir(Path(__file__).parent.parent / 'agents')
    
    from agent_state import AgentState
    from router import run_router
    from planner import run_planner
    from retriever import run_retriever
    from answerer import run_answerer
    from llm_factory import get_llm
    from mcp_client import MCPHTTPClient
    
    # Initialize dependencies
    print("\n[1] Initializing LLM and MCP client...")
    start = time.time()
    llm = get_llm()
    mcp = MCPHTTPClient()
    init_time = time.time() - start
    print(f"✓ Initialized in {init_time:.2f}s")
    
    # Create initial state
    state: AgentState = {
        "user_query": transcript,
        "agent_logs": [],
        "tool_calls": [],
        "citations": [],
        "retrieved_products": [],
    }
    
    # Profile Router
    print("\n[2] Running Router Agent...")
    start = time.time()
    state = run_router(state, llm)
    router_time = time.time() - start
    print(f"✓ Router completed in {router_time:.2f}s")
    
    # Profile Planner
    print("\n[3] Running Planner Agent...")
    start = time.time()
    state = run_planner(state)
    planner_time = time.time() - start
    print(f"✓ Planner completed in {planner_time:.2f}s")
    
    # Profile Retriever
    print("\n[4] Running Retriever Agent...")
    start = time.time()
    state = run_retriever(state, llm, mcp)
    retriever_time = time.time() - start
    print(f"✓ Retriever completed in {retriever_time:.2f}s")
    print(f"   Products retrieved: {len(state.get('retrieved_products', []))}")
    
    # Profile Answerer
    print("\n[5] Running Answerer Agent...")
    start = time.time()
    state = run_answerer(state, llm)
    answerer_time = time.time() - start
    print(f"✓ Answerer completed in {answerer_time:.2f}s")
    
    total = init_time + router_time + planner_time + retriever_time + answerer_time
    
    print(f"\n  Initialization: {init_time:.2f}s ({init_time/total*100:.1f}%)")
    print(f"  Router:         {router_time:.2f}s ({router_time/total*100:.1f}%)")
    print(f"  Planner:        {planner_time:.2f}s ({planner_time/total*100:.1f}%)")
    print(f"  Retriever:      {retriever_time:.2f}s ({retriever_time/total*100:.1f}%)")
    print(f"  Answerer:       {answerer_time:.2f}s ({answerer_time/total*100:.1f}%)")
    print(f"  ─────────────────────────────")
    print(f"  Total Agents:   {total:.2f}s")
    
    return {
        "init_time": init_time,
        "router_time": router_time,
        "planner_time": planner_time,
        "retriever_time": retriever_time,
        "answerer_time": answerer_time,
        "total": total,
        "tts_summary": state.get("tts_summary", "")
    }


def profile_tts(text):
    """Profile TTS component"""
    print("\n" + "-"*70)
    print("PROFILING: TTS (OpenAI)")
    print("-"*70)
    
    from tts import TextToSpeech
    
    # Measure initialization
    print("\n[1] Initializing TTS engine...")
    start = time.time()
    tts = TextToSpeech(
        provider="openai",
        output_dir="../test_audio",
        voice="nova"
    )
    init_time = time.time() - start
    print(f"✓ Engine initialized in {init_time:.2f}s")
    
    # Measure synthesis
    print("\n[2] Synthesizing audio...")
    print(f"   Text length: {len(text)} chars")
    start = time.time()
    audio_path = tts.synthesize_text(text, "latency_test.mp3")
    synthesis_time = time.time() - start
    print(f"✓ Synthesis in {synthesis_time:.2f}s")
    
    total = init_time + synthesis_time
    
    print(f"\n  Initialization: {init_time:.2f}s ({init_time/total*100:.1f}%)")
    print(f"  Synthesis:      {synthesis_time:.2f}s ({synthesis_time/total*100:.1f}%)")
    print(f"  ─────────────────────────────")
    print(f"  Total TTS:      {total:.2f}s")
    
    return {
        "init_time": init_time,
        "synthesis_time": synthesis_time,
        "total": total,
        "audio_path": audio_path
    }


def main():
    """Run complete profiling"""
    
    total_start = time.time()
    
    # Profile ASR
    print("\n[STEP 1/3] Profiling ASR...")
    asr_profile = profile_asr()
    
    # Profile Agents
    print("\n[STEP 2/3] Profiling Agents...")
    agents_profile = profile_agents(asr_profile["transcript"])
    
    # Profile TTS
    print("\n[STEP 3/3] Profiling TTS...")
    tts_profile = profile_tts(agents_profile["tts_summary"])
    
    total_elapsed = time.time() - total_start
    
    # Summary
    print("\n" + "="*70)
    print("LATENCY PROFILING SUMMARY")
    print("="*70)
    
    print("\n📊 Component Breakdown:")
    print(f"\n  ASR:")
    print(f"    Model Load:     {asr_profile['load_time']:.2f}s")
    print(f"    Transcription:  {asr_profile['transcribe_time']:.2f}s")
    print(f"    Subtotal:       {asr_profile['total']:.2f}s")
    
    print(f"\n  Agents:")
    print(f"    Initialization: {agents_profile['init_time']:.2f}s")
    print(f"    Router:         {agents_profile['router_time']:.2f}s")
    print(f"    Planner:        {agents_profile['planner_time']:.2f}s")
    print(f"    Retriever:      {agents_profile['retriever_time']:.2f}s")
    print(f"    Answerer:       {agents_profile['answerer_time']:.2f}s")
    print(f"    Subtotal:       {agents_profile['total']:.2f}s")
    
    print(f"\n  TTS:")
    print(f"    Initialization: {tts_profile['init_time']:.2f}s")
    print(f"    Synthesis:      {tts_profile['synthesis_time']:.2f}s")
    print(f"    Subtotal:       {tts_profile['total']:.2f}s")
    
    print(f"\n  ═════════════════════════════════════")
    print(f"  TOTAL LATENCY:  {total_elapsed:.2f}s")
    print(f"  ═════════════════════════════════════")
    
    # Bottleneck Analysis
    print("\n🔍 Bottleneck Analysis:")
    
    bottlenecks = []
    if asr_profile['load_time'] > 3:
        bottlenecks.append(f"  ⚠ ASR model loading slow ({asr_profile['load_time']:.2f}s) - consider pre-loading")
    if agents_profile['router_time'] > 5:
        bottlenecks.append(f"  ⚠ Router agent slow ({agents_profile['router_time']:.2f}s) - LLM API delay")
    if agents_profile['retriever_time'] > 5:
        bottlenecks.append(f"  ⚠ Retriever slow ({agents_profile['retriever_time']:.2f}s) - vector search or MCP delay")
    if agents_profile['answerer_time'] > 5:
        bottlenecks.append(f"  ⚠ Answerer slow ({agents_profile['answerer_time']:.2f}s) - LLM API delay")
    if tts_profile['synthesis_time'] > 5:
        bottlenecks.append(f"  ⚠ TTS synthesis slow ({tts_profile['synthesis_time']:.2f}s) - API delay")
    
    if bottlenecks:
        print("\n".join(bottlenecks))
    else:
        print("  ✓ No major bottlenecks detected")
    
    # Optimization Suggestions
    print("\n💡 Optimization Suggestions:")
    if asr_profile['load_time'] > 2:
        print(f"  • Pre-load ASR model at startup (save ~{asr_profile['load_time']:.1f}s)")
    if tts_profile['init_time'] > 1:
        print(f"  • Pre-initialize TTS engine (save ~{tts_profile['init_time']:.1f}s)")
    if agents_profile['init_time'] > 1:
        print(f"  • Cache LLM and MCP clients (save ~{agents_profile['init_time']:.1f}s)")
    
    total_savings = asr_profile['load_time'] + tts_profile['init_time'] + agents_profile['init_time']
    if total_savings > 3:
        optimized_time = total_elapsed - total_savings
        print(f"\n  Estimated with pre-loading: {optimized_time:.2f}s (save {total_savings:.2f}s)")
    
    print("\n" + "="*70 + "\n")
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
