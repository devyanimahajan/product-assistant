"""
Flask Backend API for Voice-to-Voice Product Assistant
Wraps the LangGraph workflow with REST endpoints
Supports Server-Sent Events (SSE) for real-time agent updates
"""

import os
import sys
import json
from pathlib import Path
from flask import Flask, request, jsonify, send_file, Response, stream_with_context
from flask_cors import CORS
import tempfile
import time
from datetime import datetime

# Add parent directory to path to import agents
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.resource_manager import get_resource_manager
from ASR_Zoey.asr_processor import ASRFragmentProcessor
from TTS.tts import TextToSpeech

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Initialize resource manager (pre-loads models)
print("[Backend] Initializing resource manager...")
rm = get_resource_manager()
print("[Backend] Resources loaded!")

# Audio output directory
AUDIO_DIR = Path(__file__).parent / "audio_output"
AUDIO_DIR.mkdir(exist_ok=True)


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat()
    })


@app.route('/api/query', methods=['POST'])
def process_query():
    """
    SSE endpoint: Process audio query and stream agent progress in real-time.
    
    Input: audio file (multipart/form-data)
    Output: Server-Sent Events stream with agent progress
    """
    
    def generate_events():
        temp_audio_path = None
        try:
            # 1. Get audio file from request
            if 'audio' not in request.files:
                yield f"data: {json.dumps({'type': 'error', 'message': 'No audio file provided'})}\n\n"
                return
            
            audio_file = request.files['audio']
            
            # Save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
                audio_file.save(temp_audio.name)
                temp_audio_path = temp_audio.name
            
            print(f"[Backend] Received audio file: {temp_audio_path}")
            
            # 2. Transcribe audio (ASR)
            yield f"data: {json.dumps({'type': 'phase', 'phase': 'transcribing'})}\n\n"
            
            print("[Backend] Transcribing audio...")
            asr = rm.get_asr()
            transcript_result = asr.transcribe_fragment(temp_audio_path)
            transcript = transcript_result["text"]
            print(f"[Backend] Transcript: {transcript}")
            
            yield f"data: {json.dumps({'type': 'transcript', 'text': transcript})}\n\n"
            
            # Clean up temp audio
            os.unlink(temp_audio_path)
            temp_audio_path = None
            
            # 3. Run LangGraph workflow with streaming
            print("[Backend] Running agent workflow...")
            graph = rm.get_graph()
            
            initial_state = {
                "user_query": transcript,
                "transcript": transcript,
                "agent_logs": [],
                "tool_calls": [],
                "citations": [],
                "retrieved_products": [],
            }
            
            # Use LangGraph's stream method to get updates after each node
            final_state = None
            for chunk in graph.stream(initial_state):
                # chunk is a dict with node names as keys
                for node_name, node_output in chunk.items():
                    print(f"[Backend] Node '{node_name}' completed")
                    
                    # Get the latest agent log if available
                    agent_logs = node_output.get("agent_logs", [])
                    if agent_logs:
                        latest_log = agent_logs[-1]
                        log_dict = latest_log.to_dict() if hasattr(latest_log, 'to_dict') else latest_log
                        
                        yield f"data: {json.dumps({
                            'type': 'agent_complete',
                            'agent': log_dict.get('agent_name', node_name),
                            'reasoning': log_dict.get('reasoning', ''),
                            'timestamp': log_dict.get('timestamp', '')
                        })}\n\n"
                    
                    # Store final state
                    final_state = node_output
            
            if final_state is None:
                final_state = graph.invoke(initial_state)
            
            # 4. Generate TTS audio
            yield f"data: {json.dumps({'type': 'phase', 'phase': 'tts'})}\n\n"
            
            print("[Backend] Generating TTS audio...")
            tts_summary = final_state.get("tts_summary", "")
            
            audio_url = None
            if tts_summary:
                tts = rm.get_tts()
                timestamp = int(time.time())
                audio_filename = f"response_{timestamp}.mp3"
                audio_path = str(AUDIO_DIR / audio_filename)
                tts.synthesize_text(tts_summary, audio_path)
                audio_url = f"/api/audio/{audio_filename}"
            
            # 5. Format final response
            products = []
            for product in final_state.get("retrieved_products", []):
                if hasattr(product, 'to_dict'):
                    products.append(product.to_dict())
                else:
                    products.append(product)
            
            citations = []
            for citation in final_state.get("citations", []):
                if hasattr(citation, 'to_dict'):
                    citations.append(citation.to_dict())
                else:
                    citations.append(citation)
            
            agent_logs = []
            for log in final_state.get("agent_logs", []):
                if hasattr(log, 'to_dict'):
                    agent_logs.append(log.to_dict())
                else:
                    agent_logs.append(log)
            
            # Send final complete event
            yield f"data: {json.dumps({
                'type': 'complete',
                'audio_url': audio_url,
                'final_response': final_state.get('final_response', ''),
                'tts_summary': tts_summary,
                'products': products,
                'citations': citations,
                'agent_logs': agent_logs,
                'intent': final_state.get('intent').to_dict() if final_state.get('intent') else None
            })}\n\n"
            
            print("[Backend] Request completed successfully")
            
        except Exception as e:
            print(f"[Backend] Error: {str(e)}")
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        
        finally:
            # Clean up temp file if it still exists
            if temp_audio_path and os.path.exists(temp_audio_path):
                try:
                    os.unlink(temp_audio_path)
                except:
                    pass
    
    return Response(
        stream_with_context(generate_events()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
            'Connection': 'keep-alive'
        }
    )


@app.route('/api/audio/<filename>', methods=['GET'])
def serve_audio(filename):
    """Serve TTS audio files."""
    audio_path = AUDIO_DIR / filename
    if not audio_path.exists():
        return jsonify({"error": "Audio file not found"}), 404
    
    return send_file(audio_path, mimetype='audio/mpeg')


if __name__ == '__main__':
    print("\n" + "="*60)
    print("Voice-to-Voice Product Assistant - Backend API")
    print("="*60)
    print("\nStarting server on http://localhost:5000")
    print("Endpoints:")
    print("  GET  /api/health       - Health check")
    print("  POST /api/query        - Process voice query")
    print("  GET  /api/audio/<file> - Serve TTS audio")
    print("\n" + "="*60 + "\n")
    
    app.run(host='0.0.0.0', port=5001, debug=True)
