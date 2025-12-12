import os
from typing import Dict, Any, Optional

# --- Transcription Library (Whisper) ---
try:
    # Using whisper-timestamped for easy access to segment timestamps
    import whisper_timestamped as whisper
except ImportError:
    # Define a placeholder class to prevent crash, though execution will fail
    class DummyASR:
        def __init__(self, *args, **kwargs):
            print("ASR will not function without the required library.")
        def transcribe_fragment(self, *args, **kwargs):
            return {"text": "Error: ASR library not loaded.", "segments": []}
    whisper = DummyASR()


# --- Configuration Constants ---
DEFAULT_MODEL_NAME = "small.en" 

CONTEXTUAL_PROMPT = (
    "E-commerce product recommendation. I need to buy a product. "
    "Search, find, compare, recommend, shipping, price, budget, rating, "
    "features, materials, brand, current availability, review, customer service."
)


class ASRFragmentProcessor:
    """
    Handles fragment-based Speech-to-Text using the Whisper model.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, device: str = None):
        """
        Initializes the ASR model with auto-device detection.
        
        Args:
            model_name: Whisper model to use (default: small.en for English)
            device: Device to use. If None, auto-detects: MPS → CUDA → CPU
        """
        self.model_name = model_name
        
        # Auto-detect device with priority: MPS (Apple Silicon) → CUDA (NVIDIA) → CPU
        if device is None:
            try:
                import torch
                if torch.backends.mps.is_available():
                    self.device = "mps"
                    print("[ASR] MPS (Apple Silicon GPU) detected")
                elif torch.cuda.is_available():
                    self.device = "cuda"
                    print("[ASR] CUDA (NVIDIA GPU) detected")
                else:
                    self.device = "cpu"
                    print("[ASR] Using CPU (no GPU detected)")
            except ImportError:
                self.device = "cpu"
                print("[ASR] Using CPU (torch not available)")
        else:
            self.device = device
        
        print(f"[ASR] Initializing Whisper model '{self.model_name}' on device '{self.device}'...")
        
        self.model = None
        
        if not whisper or whisper.__name__ == 'DummyASR':
            return
            
        try:
            # Load the model once during initialization
            self.model = whisper.load_model(self.model_name, device=self.device)
            print(f"[ASR] Whisper model loaded successfully on {self.device}")
        except Exception as e:
            print(f"[ASR] Error loading Whisper model: {e}")
            self.model = None

    def transcribe_fragment(self, audio_file_path: str, initial_prompt: Optional[str] = CONTEXTUAL_PROMPT) -> Dict[str, Any]:
        """
        Transcribes a single audio fragment.

        :param audio_file_path: The local path to the audio file (WAV or MP3).
        :param initial_prompt: A string to guide the model's transcription output (contextual prompting).
        :return: A dictionary containing the full 'text' transcript and a list of 'segments'.
        """
        if not self.model:
            return {"text": "ASR Model not initialized.", "segments": []}

        if not os.path.exists(audio_file_path):
            raise FileNotFoundError(f"Audio file not found at: {audio_file_path}")

        print(f"Starting transcription for {os.path.basename(audio_file_path)}...")
        
        try:
            result = whisper.transcribe(
                self.model, 
                audio_file_path, 
                language=None, 
                initial_prompt=initial_prompt,
                verbose=False,
                vad=False  # Disable voice activity detection to prevent audio trimming
            )
            
            # Extract full transcript, segment timestamps, and detected language
            transcript_data = {
                "text": result.get("text", "").strip(),
                "segments": [
                    {
                        "start": seg.get("start"),
                        "end": seg.get("end"),
                        "text": seg.get("text", "").strip(),
                    }
                    for seg in result.get("segments", [])
                ],
                "language": result.get("language")
            }
            
            print(f"Transcription successful. Detected language: {transcript_data['language']}")
            print(f"Full Transcript: \"{transcript_data['text']}\"")
            
            return transcript_data

        except Exception as e:
            print(f"An error occurred during transcription: {e}")
            return {"text": f"ASR Error: {e}", "segments": []}


# --- Example Usage (for testing) ---
if __name__ == '__main__':
    # --- DEMO CONFIGURATION ---
    test_file_path = "ASR_Zoey/test_query.mp3" 
    
    if os.path.exists(test_file_path):
        print(f"--- ASR Demonstration: Transcribing '{os.path.basename(test_file_path)}' ---")
        print(f"**Using Model:** {DEFAULT_MODEL_NAME} (General E-commerce)")
        print(f"**Contextual Prompt:** '{CONTEXTUAL_PROMPT}'")

        try:
            # Initialize the processor with the improved configuration
            asr_processor = ASRFragmentProcessor(model_name=DEFAULT_MODEL_NAME, device="cpu") 
            
            transcript_output = asr_processor.transcribe_fragment(
                test_file_path, 
                initial_prompt=CONTEXTUAL_PROMPT # Using the generalized prompt
            )
            
            print("\n=============================================")
            print(" ASR Output for LangGraph Router Input ")
            print("=============================================")
            print(f"**INPUT TEXT (to Router):** {transcript_output['text']}")
            print(f"**Detected Language:** {transcript_output['language']}")
            
            if transcript_output['segments']:
                print("\n**Segment Timestamps:**")
                for i, segment in enumerate(transcript_output['segments']):
                    print(f"  [{i+1}] {segment['start']:.2f}s - {segment['end']:.2f}s: {segment['text']}")

        except Exception as e:
            print(f"\nA general error occurred in the ASR demo: {e}")

    else:
        print(f"\n[DEMO SKIPPED] Please ensure the test file exists at: {test_file_path}")

# Integration Notes

# UI Integration: In your Streamlit UI component, you will implement the logic to:
# Capture/upload a .wav or .mp3 file (the "fragment").
# Save the uploaded file temporarily.
# Instantiate ASRFragmentProcessor and call transcribe_fragment(path_to_file).
# The resulting transcript_output dictionary will be the primary input to the Agentic Orchestration (LangGraph) system's first node (the Router/Intent Classifier).

# LangGraph Input: The Router agent in your LangGraph will receive the string: transcript_output["text"]. The timestamps (transcript_output["segments"]) can be used for UI display or for future alignment/correction mechanisms but are not strictly required for the immediate agent task.
