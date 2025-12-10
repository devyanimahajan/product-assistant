"""
Speech Recognition (ASR) Tool - Convert speech to text using Whisper

Supports both fragment-based (WAV/MP3 upload) and streaming audio input.
Returns accurate transcripts with optional timestamps for multilingual accents.
"""

import os
import wave
import tempfile
import numpy as np
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
from datetime import datetime
import io

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("Warning: openai-whisper not installed. Install with: pip install openai-whisper")

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    print("Warning: pydub not installed. Install with: pip install pydub")


# Global model cache for lazy loading
_whisper_model = None
_model_size = None


def get_whisper_model(model_size: str = "base"):
    """
    Lazy load Whisper model with caching.
    
    Available models:
    - tiny: Fastest, less accurate (~1GB RAM)
    - base: Balanced speed/accuracy (~1GB RAM) - DEFAULT
    - small: Better accuracy (~2GB RAM)
    - medium: High accuracy (~5GB RAM)
    - large: Best accuracy (~10GB RAM)
    
    Args:
        model_size: Model size to load
    
    Returns:
        Loaded Whisper model
    """
    global _whisper_model, _model_size
    
    if not WHISPER_AVAILABLE:
        raise ImportError(
            "openai-whisper is required. Install with: pip install openai-whisper"
        )
    
    if _whisper_model is None or _model_size != model_size:
        print(f"[asr] Loading Whisper model '{model_size}'...")
        _whisper_model = whisper.load_model(model_size)
        _model_size = model_size
        print(f"[asr] Model loaded successfully")
    
    return _whisper_model


def convert_to_wav(audio_path: str, output_path: Optional[str] = None) -> str:
    """
    Convert audio file to WAV format (16kHz, mono) if needed.
    
    Supports: MP3, M4A, FLAC, OGG, WAV, WEBM
    
    Args:
        audio_path: Path to input audio file
        output_path: Optional output path (creates temp file if None)
    
    Returns:
        Path to WAV file
    """
    if not PYDUB_AVAILABLE:
        # If pydub not available, assume input is already WAV
        print("[asr] pydub not available, assuming input is WAV format")
        return audio_path
    
    audio_path = Path(audio_path)
    
    # Check if already WAV with correct format
    if audio_path.suffix.lower() == '.wav':
        try:
            with wave.open(str(audio_path), 'rb') as wf:
                if wf.getnchannels() == 1 and wf.getframerate() == 16000:
                    return str(audio_path)
        except Exception:
            pass  # Conversion needed
    
    print(f"[asr] Converting {audio_path.suffix} to WAV (16kHz mono)...")
    
    try:
        # Load audio file
        audio = AudioSegment.from_file(str(audio_path))
        
        # Convert to mono and 16kHz
        audio = audio.set_channels(1)
        audio = audio.set_frame_rate(16000)
        
        # Export as WAV
        if output_path is None:
            output_path = tempfile.NamedTemporaryFile(
                delete=False, suffix='.wav'
            ).name
        
        audio.export(output_path, format='wav')
        print(f"[asr] Converted to: {output_path}")
        
        return output_path
        
    except Exception as e:
        print(f"[asr] Conversion failed: {e}")
        return str(audio_path)  # Return original if conversion fails


def transcribe_audio(
    audio_path: str,
    model_size: str = "base",
    language: Optional[str] = None,
    task: str = "transcribe",
    return_timestamps: bool = True,
    word_timestamps: bool = False
) -> Dict[str, Any]:
    """
    Transcribe audio file using Whisper.
    
    Args:
        audio_path: Path to audio file (WAV, MP3, M4A, etc.)
        model_size: Whisper model size (tiny/base/small/medium/large)
        language: Force specific language (e.g., 'en', 'zh', 'es')
                 None = auto-detect
        task: 'transcribe' (keep original language) or 'translate' (to English)
        return_timestamps: Include segment timestamps
        word_timestamps: Include word-level timestamps (slower, requires alignment)
    
    Returns:
        {
            "success": bool,
            "text": str,              # Full transcript
            "language": str,          # Detected/specified language
            "segments": [             # Optional: timestamped segments
                {
                    "id": int,
                    "start": float,   # seconds
                    "end": float,     # seconds
                    "text": str
                }
            ],
            "duration": float,        # Audio duration in seconds
            "model_size": str
        }
    """
    try:
        # Validate audio file exists
        if not os.path.exists(audio_path):
            return {
                "success": False,
                "error": f"Audio file not found: {audio_path}",
                "text": ""
            }
        
        # Convert to WAV if needed
        wav_path = convert_to_wav(audio_path)
        
        # Load Whisper model
        model = get_whisper_model(model_size)
        
        # Transcription options
        options = {
            "task": task,
            "language": language,
            "verbose": False
        }
        
        if word_timestamps:
            options["word_timestamps"] = True
        
        print(f"[asr] Transcribing audio: {Path(audio_path).name}")
        print(f"[asr] Model: {model_size}, Language: {language or 'auto-detect'}, Task: {task}")
        
        # Transcribe
        result = model.transcribe(wav_path, **options)
        
        # Build response
        response = {
            "success": True,
            "text": result["text"].strip(),
            "language": result.get("language", "unknown"),
            "model_size": model_size,
            "task": task
        }
        
        # Add timestamps if requested
        if return_timestamps and "segments" in result:
            segments = []
            for seg in result["segments"]:
                segment_data = {
                    "id": seg["id"],
                    "start": round(seg["start"], 2),
                    "end": round(seg["end"], 2),
                    "text": seg["text"].strip()
                }
                
                # Add word-level timestamps if available
                if word_timestamps and "words" in seg:
                    segment_data["words"] = [
                        {
                            "word": w["word"],
                            "start": round(w["start"], 2),
                            "end": round(w["end"], 2)
                        }
                        for w in seg["words"]
                    ]
                
                segments.append(segment_data)
            
            response["segments"] = segments
            
            # Add duration
            if segments:
                response["duration"] = round(segments[-1]["end"], 2)
        
        # Clean up temp WAV file if created
        if wav_path != audio_path and os.path.exists(wav_path):
            try:
                os.remove(wav_path)
            except Exception:
                pass
        
        print(f"[asr] Transcription complete: {len(response['text'])} characters")
        
        return response
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Transcription failed: {str(e)}",
            "text": ""
        }


def transcribe_audio_stream(
    audio_chunks: List[bytes],
    model_size: str = "base",
    language: Optional[str] = None,
    sample_rate: int = 16000,
    return_timestamps: bool = False
) -> Dict[str, Any]:
    """
    Transcribe audio from streaming chunks (for real-time applications).
    
    NOTE: This concatenates chunks into a single audio buffer before transcription.
    For true streaming ASR, consider using faster-whisper or whisper-streaming.
    
    Args:
        audio_chunks: List of audio byte chunks (raw PCM or WAV)
        model_size: Whisper model size
        language: Force specific language (None = auto-detect)
        sample_rate: Audio sample rate in Hz
        return_timestamps: Include segment timestamps
    
    Returns:
        Same format as transcribe_audio()
    """
    try:
        # Concatenate audio chunks
        audio_bytes = b''.join(audio_chunks)
        
        # Create temporary WAV file
        with tempfile.NamedTemporaryFile(
            delete=False, suffix='.wav'
        ) as temp_file:
            temp_path = temp_file.name
            
            # Write as WAV
            with wave.open(temp_path, 'wb') as wf:
                wf.setnchannels(1)  # Mono
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(sample_rate)
                wf.writeframes(audio_bytes)
        
        # Transcribe using fragment-based method
        result = transcribe_audio(
            audio_path=temp_path,
            model_size=model_size,
            language=language,
            return_timestamps=return_timestamps
        )
        
        # Clean up temp file
        try:
            os.remove(temp_path)
        except Exception:
            pass
        
        return result
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Stream transcription failed: {str(e)}",
            "text": ""
        }


def batch_transcribe(
    audio_files: List[str],
    model_size: str = "base",
    language: Optional[str] = None,
    parallel: bool = False
) -> List[Dict[str, Any]]:
    """
    Batch transcribe multiple audio files.
    
    Args:
        audio_files: List of paths to audio files
        model_size: Whisper model size
        language: Force specific language
        parallel: Use parallel processing (requires multiprocessing)
    
    Returns:
        List of transcription results
    """
    results = []
    
    if parallel:
        try:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor() as executor:
                futures = [
                    executor.submit(
                        transcribe_audio,
                        audio_path=path,
                        model_size=model_size,
                        language=language
                    )
                    for path in audio_files
                ]
                results = [f.result() for f in futures]
        except Exception as e:
            print(f"[asr] Parallel processing failed: {e}, falling back to sequential")
            parallel = False
    
    if not parallel:
        for i, audio_path in enumerate(audio_files, 1):
            print(f"\n[asr] Processing {i}/{len(audio_files)}: {Path(audio_path).name}")
            result = transcribe_audio(
                audio_path=audio_path,
                model_size=model_size,
                language=language
            )
            results.append(result)
    
    return results


def get_supported_languages() -> List[str]:
    """
    Get list of languages supported by Whisper.
    
    Returns:
        List of language codes (e.g., ['en', 'zh', 'es', 'fr', ...])
    """
    if not WHISPER_AVAILABLE:
        return []
    
    # Whisper supports 99 languages
    return list(whisper.tokenizer.LANGUAGES.keys())


def detect_language(audio_path: str, model_size: str = "base") -> Dict[str, Any]:
    """
    Detect language of audio file without full transcription.
    
    Args:
        audio_path: Path to audio file
        model_size: Whisper model size
    
    Returns:
        {
            "success": bool,
            "language": str,      # Language code (e.g., 'en')
            "language_name": str, # Language name (e.g., 'English')
            "confidence": float   # Detection confidence (0-1)
        }
    """
    try:
        model = get_whisper_model(model_size)
        
        # Load audio
        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)
        
        # Make log-Mel spectrogram
        mel = whisper.log_mel_spectrogram(audio).to(model.device)
        
        # Detect language
        _, probs = model.detect_language(mel)
        
        # Get top language
        detected_lang = max(probs, key=probs.get)
        confidence = probs[detected_lang]
        
        return {
            "success": True,
            "language": detected_lang,
            "language_name": whisper.tokenizer.LANGUAGES.get(
                detected_lang, detected_lang
            ),
            "confidence": round(confidence, 3),
            "all_probabilities": {
                lang: round(prob, 3)
                for lang, prob in sorted(
                    probs.items(), key=lambda x: x[1], reverse=True
                )[:5]  # Top 5
            }
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Language detection failed: {str(e)}",
            "language": "unknown"
        }


# Example usage and testing
if __name__ == "__main__":
    print("\n=== ASR TOOL TEST ===\n")
    
    # Check if Whisper is available
    if not WHISPER_AVAILABLE:
        print("ERROR: openai-whisper is not installed")
        print("Install with: pip install openai-whisper")
        exit(1)
    
    # Print supported languages
    languages = get_supported_languages()
    print(f"Supported languages: {len(languages)}")
    print(f"Sample: {languages[:10]}")
    
    # Test with sample audio file (if exists)
    sample_audio = "sample_audio.wav"
    
    if os.path.exists(sample_audio):
        print(f"\n--- Testing with {sample_audio} ---\n")
        
        # Language detection
        lang_result = detect_language(sample_audio)
        print(f"Detected language: {lang_result}")
        
        # Transcription
        result = transcribe_audio(
            audio_path=sample_audio,
            model_size="base",
            return_timestamps=True
        )
        
        if result["success"]:
            print(f"\nTranscript ({result['language']}):")
            print(result["text"])
            
            if "segments" in result:
                print(f"\nSegments ({len(result['segments'])}):")
                for seg in result["segments"][:3]:
                    print(f"  [{seg['start']:.1f}s - {seg['end']:.1f}s] {seg['text']}")
        else:
            print(f"ERROR: {result['error']}")
    else:
        print(f"\nNo sample audio file found at: {sample_audio}")
        print("Place a WAV/MP3 file named 'sample_audio.wav' to test")
    
    print("\n=== ASR TOOL READY ===")
