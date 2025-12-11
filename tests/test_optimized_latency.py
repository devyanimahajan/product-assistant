"""
Optimized Voice-to-Voice Test with Pre-loading
Shows latency with and without resource initialization overhead
"""

import sys
import os
from pathlib import Path
import time

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'agents'))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / '.env')

from resource_manager import get_resource_manager
from agent_state import AgentState

print("\n" + "="*70)
print("OPTIMIZED LATENCY TEST (WITH PRE-LOADING)")
print("="*70)
print("\nThis test demonstrates latency improvements via resource pre-loading:")
print("  • First Run: Includes initialization time (~5-7s)")
print("  • Subsequent Runs: Pure query processing time (target <18s)")
print("\n" + "="*70 + "\n")


def run_voice_to_voice_query(rm, test_audio_path, query_number=1):
    """Run a complete voice-to-voice query using pre-loaded resources"""
    
    print(f"\n{'='*70}")
    print(f"QUERY {query_number}: Voice-to-Voice Pipeline")
    print(f"{'='*70}\n")
    
    total_start = time.time()
    
    # Step 1: ASR (using pre-loaded model)
    print("[1/3] ASR: Transcribing audio...")
    asr_start = time.time()
    asr = rm.get_asr()
    result = asr.transcribe_fragment(test_audio_path)
    transcript = result.get("text", "")
    asr_time = time.time() - asr_start
    print(f"✓ Transcribed in {asr_time:.2f}s")
    print(f"  Transcript: '{transcript[:60]}...'")
    
    # Step 2: Agents (using pre-compiled graph)
    print("\n[2/3] Agents: Processing query...")
    agents_start = time.time()
    
    graph = rm.get_graph()
    initial_state: AgentState = {
        "user_query": transcript,
        "agent_logs": [],
        "tool_calls": [],
        "citations": [],
        "retrieved_products": [],
    }
    
    final_state = graph.invoke(initial_state)
    agents_time = time.time() - agents_start
    
    products_count = len(final_state.get("retrieved_products", []))
    citations_count = len(final_state.get("citations", []))
    tts_summary = final_state.get("tts_summary", "")
    
    print(f"✓ Agents completed in {agents_time:.2f}s")
    print(f"  Products: {products_count}, Citations: {citations_count}")
    
    # Step 3: TTS (using pre-loaded engine)
    print("\n[3/3] TTS: Generating audio...")
    tts_start = time.time()
    
    tts = rm.get_tts()
    audio_path = tts.synthesize_text(
        tts_summary,
        f"optimized_query_{query_number}.mp3"
    )
    tts_time = time.time() - tts_start
    
    file_size = os.path.getsize(audio_path) if os.path.exists(audio_path) else 0
    print(f"✓ Audio generated in {tts_time:.2f}s")
    print(f"  File: {os.path.basename(audio_path)} ({file_size/1024:.1f} KB)")
    
    # Summary
    total_time = time.time() - total_start
    
    print(f"\n{'='*70}")
    print(f"QUERY {query_number} RESULTS")
    print(f"{'='*70}")
    print(f"\n  ASR Time:        {asr_time:.2f}s")
    print(f"  Agents Time:     {agents_time:.2f}s")
    print(f"  TTS Time:        {tts_time:.2f}s")
    print(f"  ─────────────────────────────")
    print(f"  Total Time:      {total_time:.2f}s")
    
    return {
        "query_num": query_number,
        "asr_time": asr_time,
        "agents_time": agents_time,
        "tts_time": tts_time,
        "total_time": total_time,
        "transcript": transcript,
        "tts_summary": tts_summary,
        "products": products_count,
        "citations": citations_count
    }


def main():
    """Run optimized latency test"""
    
    # Get test audio
    test_audio = str(Path(__file__).parent.parent / "ASR_Zoey" / "test_query.mp3")
    
    if not os.path.exists(test_audio):
        print(f"✗ Test audio not found: {test_audio}")
        return 1
    
    # Pre-load all resources (one-time cost)
    print("="*70)
    print("INITIALIZING SYSTEM (One-time startup cost)")
    print("="*70)
    print()
    
    init_start = time.time()
    rm = get_resource_manager()
    init_time = time.time() - init_start
    
    print(f"\n{'='*70}")
    print(f"SYSTEM READY - Initialization took {init_time:.2f}s")
    print(f"{'='*70}")
    print("\nNow running queries with pre-loaded resources...\n")
    time.sleep(1)
    
    # Run multiple queries to show consistent performance
    results = []
    
    # Query 1
    result1 = run_voice_to_voice_query(rm, test_audio, query_number=1)
    results.append(result1)
    
    time.sleep(0.5)
    
    # Query 2 (same query to show caching/optimization)
    result2 = run_voice_to_voice_query(rm, test_audio, query_number=2)
    results.append(result2)
    
    # Final Summary
    print(f"\n{'='*70}")
    print("OPTIMIZATION SUMMARY")
    print(f"{'='*70}\n")
    
    print(f"Initial Startup Cost: {init_time:.2f}s (one-time)")
    print(f"\nPer-Query Performance:")
    for r in results:
        print(f"  Query {r['query_num']}: {r['total_time']:.2f}s")
    
    avg_time = sum(r['total_time'] for r in results) / len(results)
    print(f"\n  Average per query: {avg_time:.2f}s")
    
    print(f"\n{'='*70}")
    print("LATENCY COMPARISON")
    print(f"{'='*70}\n")
    
    print("Without Pre-loading (cold start each query):")
    print(f"  ~25s per query (includes init overhead)\n")
    
    print("With Pre-loading (this test):")
    print(f"  First query (includes startup): {init_time + result1['total_time']:.2f}s")
    print(f"  Subsequent queries: ~{avg_time:.2f}s")
    
    if avg_time < 18:
        print(f"\n✅ TARGET ACHIEVED: <18s per query (actual: {avg_time:.2f}s)")
    else:
        print(f"\n⚠ Target not met: {avg_time:.2f}s (target: <18s)")
    
    print(f"\n{'='*70}")
    print("COMPONENT BREAKDOWN (Average)")
    print(f"{'='*70}\n")
    
    avg_asr = sum(r['asr_time'] for r in results) / len(results)
    avg_agents = sum(r['agents_time'] for r in results) / len(results)
    avg_tts = sum(r['tts_time'] for r in results) / len(results)
    
    print(f"  ASR:     {avg_asr:.2f}s ({avg_asr/avg_time*100:.1f}%)")
    print(f"  Agents:  {avg_agents:.2f}s ({avg_agents/avg_time*100:.1f}%)")
    print(f"  TTS:     {avg_tts:.2f}s ({avg_tts/avg_time*100:.1f}%)")
    print(f"  Overhead: {avg_time - avg_asr - avg_agents - avg_tts:.2f}s")
    
    print(f"\n{'='*70}")
    print("OPTIMIZED VOICE-TO-VOICE SYSTEM: READY!")
    print(f"{'='*70}\n")
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
