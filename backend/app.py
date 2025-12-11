"""
Flask Backend API for Voice-to-Voice Product Assistant
Wraps the LangGraph workflow with REST endpoints
"""

import os
import sys
from pathlib import Path
from flask import Flask, request, jsonify, send_file
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
    Main endpoint: Process audio query end-to-end.
    
    Input: audio file (multipart/form-data)
    Output: JSON with transcript, products, agent logs, TTS audio URL
    """
    try:
        # 1. Get audio file from request
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400
        
        audio_file = request.files['audio']
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
            audio_file.save(temp_audio.name)
            temp_audio_path = temp_audio.name
        
        print(f"[Backend] Received audio file: {temp_audio_path}")
        
        # 2. Transcribe audio (ASR)
        print("[Backend] Transcribing audio...")
        asr = rm.get_asr()
        transcript_result = asr.transcribe_fragment(temp_audio_path)
        transcript = transcript_result["text"]
        print(f"[Backend] Transcript: {transcript}")
        
        # Clean up temp audio
        os.unlink(temp_audio_path)
        
        # 3. Run LangGraph workflow
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
        
        final_state = graph.invoke(initial_state)
        
        # 4. Generate TTS audio
        print("[Backend] Generating TTS audio...")
        tts_summary = final_state.get("tts_summary", "")
        
        if tts_summary:
            tts = rm.get_tts()
            timestamp = int(time.time())
            audio_filename = f"response_{timestamp}.mp3"
            audio_path = str(AUDIO_DIR / audio_filename)
            tts.synthesize_text(tts_summary, audio_path)
            audio_url = f"/api/audio/{audio_filename}"
        else:
            audio_url = None
        
        # 5. Format response
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
        
        response = {
            "transcript": transcript,
            "final_response": final_state.get("final_response", ""),
            "tts_summary": tts_summary,
            "audio_url": audio_url,
            "products": products,
            "citations": citations,
            "agent_logs": agent_logs,
            "intent": final_state.get("intent").to_dict() if final_state.get("intent") else None,
        }
        
        print("[Backend] Request completed successfully")
        return jsonify(response)
    
    except Exception as e:
        print(f"[Backend] Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


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
