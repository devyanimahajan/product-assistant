"""
ASR API Endpoints - Flask integration for speech recognition

Provides REST endpoints for:
- /asr/transcribe - Upload audio file for transcription
- /asr/detect-language - Detect audio language
- /asr/health - Check ASR system status
"""

import os
import tempfile
from pathlib import Path
from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename

from asr_tool import (
    transcribe_audio,
    detect_language,
    get_supported_languages,
    WHISPER_AVAILABLE
)

# Create Blueprint
asr_bp = Blueprint('asr', __name__, url_prefix='/asr')

# Allowed audio file extensions
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a', 'flac', 'ogg', 'webm'}

# Maximum file size (50 MB)
MAX_FILE_SIZE = 50 * 1024 * 1024


def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@asr_bp.route('/health', methods=['GET'])
def health_check():
    """
    Check ASR system health and capabilities.
    
    Returns:
        {
            "status": "healthy" | "degraded",
            "whisper_available": bool,
            "supported_languages_count": int,
            "allowed_formats": list,
            "max_file_size_mb": int
        }
    """
    return jsonify({
        "status": "healthy" if WHISPER_AVAILABLE else "degraded",
        "whisper_available": WHISPER_AVAILABLE,
        "supported_languages_count": len(get_supported_languages()) if WHISPER_AVAILABLE else 0,
        "allowed_formats": list(ALLOWED_EXTENSIONS),
        "max_file_size_mb": MAX_FILE_SIZE // (1024 * 1024)
    })


@asr_bp.route('/languages', methods=['GET'])
def list_languages():
    """
    Get list of supported languages.
    
    Returns:
        {
            "success": bool,
            "languages": [str],  # Language codes
            "count": int
        }
    """
    if not WHISPER_AVAILABLE:
        return jsonify({
            "success": False,
            "error": "Whisper is not available",
            "languages": []
        }), 503
    
    languages = get_supported_languages()
    
    return jsonify({
        "success": True,
        "languages": languages,
        "count": len(languages)
    })


@asr_bp.route('/transcribe', methods=['POST'])
def transcribe():
    """
    Transcribe uploaded audio file.
    
    Request:
        - Form data with 'audio' file
        - Optional form fields:
          - model_size: str (tiny/base/small/medium/large)
          - language: str (force language, e.g., 'en', 'zh')
          - task: str (transcribe/translate)
          - timestamps: bool (include segment timestamps)
          - word_timestamps: bool (word-level timestamps)
    
    Returns:
        {
            "success": bool,
            "text": str,
            "language": str,
            "segments": [...],  # Optional
            "duration": float,  # Optional
            "error": str  # On failure
        }
    """
    if not WHISPER_AVAILABLE:
        return jsonify({
            "success": False,
            "error": "Whisper is not available. Install with: pip install openai-whisper"
        }), 503
    
    # Check if file is present
    if 'audio' not in request.files:
        return jsonify({
            "success": False,
            "error": "No audio file provided"
        }), 400
    
    file = request.files['audio']
    
    # Check if file has a name
    if file.filename == '':
        return jsonify({
            "success": False,
            "error": "Empty filename"
        }), 400
    
    # Check file extension
    if not allowed_file(file.filename):
        return jsonify({
            "success": False,
            "error": f"Invalid file format. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        }), 400
    
    # Get optional parameters
    model_size = request.form.get('model_size', 'base')
    language = request.form.get('language', None)
    task = request.form.get('task', 'transcribe')
    return_timestamps = request.form.get('timestamps', 'true').lower() == 'true'
    word_timestamps = request.form.get('word_timestamps', 'false').lower() == 'true'
    
    # Validate model size
    valid_models = ['tiny', 'base', 'small', 'medium', 'large']
    if model_size not in valid_models:
        return jsonify({
            "success": False,
            "error": f"Invalid model_size. Choose from: {', '.join(valid_models)}"
        }), 400
    
    # Validate task
    if task not in ['transcribe', 'translate']:
        return jsonify({
            "success": False,
            "error": "Invalid task. Choose: 'transcribe' or 'translate'"
        }), 400
    
    try:
        # Save uploaded file temporarily
        filename = secure_filename(file.filename)
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, f"asr_{os.getpid()}_{filename}")
        
        file.save(temp_path)
        
        # Check file size
        file_size = os.path.getsize(temp_path)
        if file_size > MAX_FILE_SIZE:
            os.remove(temp_path)
            return jsonify({
                "success": False,
                "error": f"File too large. Maximum: {MAX_FILE_SIZE // (1024*1024)} MB"
            }), 400
        
        # Transcribe
        result = transcribe_audio(
            audio_path=temp_path,
            model_size=model_size,
            language=language if language else None,
            task=task,
            return_timestamps=return_timestamps,
            word_timestamps=word_timestamps
        )
        
        # Clean up temp file
        try:
            os.remove(temp_path)
        except Exception:
            pass
        
        # Return result
        status_code = 200 if result["success"] else 500
        return jsonify(result), status_code
        
    except Exception as e:
        # Clean up on error
        if 'temp_path' in locals():
            try:
                os.remove(temp_path)
            except Exception:
                pass
        
        return jsonify({
            "success": False,
            "error": f"Transcription failed: {str(e)}"
        }), 500


@asr_bp.route('/detect-language', methods=['POST'])
def detect_lang():
    """
    Detect language of uploaded audio file.
    
    Request:
        - Form data with 'audio' file
        - Optional: model_size (tiny/base/small/medium/large)
    
    Returns:
        {
            "success": bool,
            "language": str,
            "language_name": str,
            "confidence": float,
            "all_probabilities": dict,
            "error": str  # On failure
        }
    """
    if not WHISPER_AVAILABLE:
        return jsonify({
            "success": False,
            "error": "Whisper is not available"
        }), 503
    
    # Check if file is present
    if 'audio' not in request.files:
        return jsonify({
            "success": False,
            "error": "No audio file provided"
        }), 400
    
    file = request.files['audio']
    
    if file.filename == '':
        return jsonify({
            "success": False,
            "error": "Empty filename"
        }), 400
    
    if not allowed_file(file.filename):
        return jsonify({
            "success": False,
            "error": f"Invalid file format. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        }), 400
    
    model_size = request.form.get('model_size', 'base')
    
    try:
        # Save uploaded file temporarily
        filename = secure_filename(file.filename)
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, f"asr_lang_{os.getpid()}_{filename}")
        
        file.save(temp_path)
        
        # Detect language
        result = detect_language(temp_path, model_size=model_size)
        
        # Clean up temp file
        try:
            os.remove(temp_path)
        except Exception:
            pass
        
        status_code = 200 if result["success"] else 500
        return jsonify(result), status_code
        
    except Exception as e:
        # Clean up on error
        if 'temp_path' in locals():
            try:
                os.remove(temp_path)
            except Exception:
                pass
        
        return jsonify({
            "success": False,
            "error": f"Language detection failed: {str(e)}"
        }), 500


# Optional: Register blueprint with app
def register_asr_routes(app):
    """
    Register ASR routes with Flask app.
    
    Usage:
        from asr_api import register_asr_routes
        app = Flask(__name__)
        register_asr_routes(app)
    """
    app.register_blueprint(asr_bp)
    print("[ASR API] Routes registered:")
    print("  - POST /asr/transcribe")
    print("  - POST /asr/detect-language")
    print("  - GET  /asr/languages")
    print("  - GET  /asr/health")
