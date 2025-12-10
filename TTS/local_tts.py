"""
Text-to-Speech module using Microsoft SpeechT5 (Open-source, Local)
Based on the Speech_to_Speech_LLM_System notebook.

This is a FREE, LOCAL alternative to commercial TTS APIs.
Suitable for offline environments and budget-conscious projects.
"""

import torch
import soundfile as sf
import numpy as np
from pathlib import Path
from typing import Optional, Literal
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

try:
    from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
    from datasets import load_dataset
    SPEECHT5_AVAILABLE = True
except ImportError:
    SPEECHT5_AVAILABLE = False


class SpeechT5TTS:
    """
    SpeechT5-based Text-to-Speech engine.
    
    Features:
    - Open-source and free
    - Runs locally (no API key needed)
    - Offline capable
    - Decent quality (not as good as commercial APIs)
    
    Requirements:
    - GPU recommended (but works on CPU)
    - ~400MB model download on first run
    """
    
    def __init__(
        self,
        speaker_idx: int = 7306,
        device: Optional[str] = None,
        output_dir: str = "./audio_output"
    ):
        """
        Initialize SpeechT5 TTS.
        
        Args:
            speaker_idx: Speaker voice ID (0-7000+)
            device: Device to use ('cuda', 'cpu', or None for auto)
            output_dir: Directory to save audio files
        """
        if not SPEECHT5_AVAILABLE:
            raise ImportError(
                "SpeechT5 dependencies not installed. Run:\n"
                "pip install transformers datasets soundfile torch torchaudio"
            )
        
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"[SpeechT5] Using device: {self.device}")
        
        if self.device == "cpu":
            print("[SpeechT5] WARNING: Running on CPU. Synthesis will be slower.")
            print("[SpeechT5] For faster performance, use a GPU.")
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load models
        print("[SpeechT5] Loading models (first time will download ~400MB)...")
        
        self.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
        self.model = SpeechT5ForTextToSpeech.from_pretrained(
            "microsoft/speecht5_tts"
        ).to(self.device)
        self.vocoder = SpeechT5HifiGan.from_pretrained(
            "microsoft/speecht5_hifigan"
        ).to(self.device)
        
        # Load speaker embeddings
        print("[SpeechT5] Loading speaker embeddings...")
        try:
            embeddings_dataset = load_dataset(
                "Matthijs/cmu-arctic-xvectors",
                split="validation",
                trust_remote_code=True
            )
            
            self.speaker_embeddings = torch.tensor(
                embeddings_dataset[speaker_idx]["xvector"]
            ).unsqueeze(0).to(self.device)
            
            print(f"[SpeechT5] Using speaker voice #{speaker_idx}")
            
        except Exception as e:
            print(f"[SpeechT5] WARNING: Failed to load speaker embeddings: {e}")
            print("[SpeechT5] Using default embeddings")
            # Fallback to random embeddings
            self.speaker_embeddings = torch.randn(1, 512).to(self.device)
        
        print("[SpeechT5] Initialization complete!")
    
    def synthesize(
        self,
        text: str,
        output_path: Optional[str] = None,
        max_length: int = 1200
    ) -> str:
        """
        Synthesize text to speech.
        
        Args:
            text: Text to synthesize
            output_path: Path to save audio file (auto-generated if None)
            max_length: Maximum text length (longer text will be truncated)
            
        Returns:
            Path to generated audio file
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        # Truncate if too long
        if len(text) > max_length:
            truncate_pos = text[:max_length].rfind('.')
            if truncate_pos == -1:
                truncate_pos = max_length
            text = text[:truncate_pos + 1]
            print(f"[SpeechT5] Text truncated to {len(text)} characters")
        
        # Generate output path if not provided
        if output_path is None:
            import time
            timestamp = int(time.time())
            output_path = str(self.output_dir / f"speech_{timestamp}.wav")
        
        print(f"[SpeechT5] Synthesizing {len(text)} characters...")
        
        # Process text
        inputs = self.processor(text=text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        
        # Generate speech
        with torch.no_grad():
            speech = self.model.generate_speech(
                input_ids,
                self.speaker_embeddings,
                vocoder=self.vocoder
            )
        
        # Save audio file
        speech_numpy = speech.cpu().numpy()
        sf.write(output_path, speech_numpy, samplerate=16000)
        
        print(f"[SpeechT5] Audio saved to: {output_path}")
        
        return output_path
    
    def generate_summary(
        self,
        product_recommendation: dict,
        max_length: int = 150
    ) -> str:
        """
        Generate a concise spoken summary from product recommendation.
        
        Args:
            product_recommendation: Dict with product info
            max_length: Max characters for summary
            
        Returns:
            Summary text suitable for TTS
        """
        title = product_recommendation.get('title', 'Product')
        price = product_recommendation.get('price', 0)
        rating = product_recommendation.get('rating', 0)
        brand = product_recommendation.get('brand', '')
        
        # Build summary
        summary_parts = []
        
        # Main recommendation
        if brand:
            summary_parts.append(f"I recommend {brand} {title}")
        else:
            summary_parts.append(f"I recommend {title}")
        
        # Price
        summary_parts.append(f"priced at ${price:.2f}")
        
        # Rating
        if rating > 0:
            summary_parts.append(f"with a {rating:.1f} star rating")
        
        summary = ", ".join(summary_parts) + "."
        
        # Add alternatives mention
        alternatives_count = product_recommendation.get('alternatives_count', 0)
        if alternatives_count > 0:
            summary += f" I found {alternatives_count} similar options. Check your screen for details."
        
        # Truncate if too long
        if len(summary) > max_length:
            summary = summary[:max_length-3] + "..."
        
        return summary
    
    def synthesize_recommendation(
        self,
        product_recommendation: dict,
        filename: Optional[str] = None
    ) -> tuple[str, str]:
        """
        Generate and synthesize speech for product recommendation.
        
        Args:
            product_recommendation: Product recommendation dict
            filename: Optional output filename
            
        Returns:
            Tuple of (summary_text, audio_file_path)
        """
        summary = self.generate_summary(product_recommendation)
        audio_path = self.synthesize(summary, filename)
        return summary, audio_path


class LocalTTS:
    """
    Wrapper class compatible with the main TTS interface.
    Provides the same API as the commercial TTS providers.
    """
    
    def __init__(
        self,
        output_dir: str = "./audio_output",
        speaker_idx: int = 7306,
        **kwargs
    ):
        """
        Initialize local TTS.
        
        Args:
            output_dir: Directory to save audio files
            speaker_idx: Speaker voice ID
        """
        self.provider = SpeechT5TTS(
            speaker_idx=speaker_idx,
            output_dir=output_dir
        )
        self.output_dir = Path(output_dir)
        self.file_extension = "wav"
    
    def synthesize_text(
        self,
        text: str,
        filename: Optional[str] = None
    ) -> str:
        """Synthesize text to speech."""
        return self.provider.synthesize(text, filename)
    
    def generate_summary(
        self,
        product_recommendation: dict,
        max_length: int = 150
    ) -> str:
        """Generate product summary."""
        return self.provider.generate_summary(product_recommendation, max_length)
    
    def synthesize_recommendation(
        self,
        product_recommendation: dict,
        filename: Optional[str] = None
    ) -> tuple[str, str]:
        """Generate and synthesize product recommendation."""
        return self.provider.synthesize_recommendation(product_recommendation, filename)


# Factory function for easy switching
def get_local_tts(**kwargs) -> LocalTTS:
    """
    Get local TTS engine.
    
    Usage:
        tts = get_local_tts()
        audio_path = tts.synthesize_text("Hello world")
    """
    return LocalTTS(**kwargs)


# Example usage
if __name__ == "__main__":
    print("\n=== TESTING LOCAL TTS (SpeechT5) ===\n")
    
    try:
        # Initialize TTS
        print("Initializing SpeechT5 TTS...")
        tts = LocalTTS(output_dir="./test_audio")
        
        # Test 1: Simple text
        print("\nTest 1: Simple text synthesis")
        text = "Welcome to the product assistant. This is a test of the local text to speech system."
        audio_path = tts.synthesize_text(text, "test_simple.wav")
        print(f"✓ Audio generated: {audio_path}")
        
        # Test 2: Product recommendation
        print("\nTest 2: Product recommendation")
        product = {
            'title': 'Eco-Friendly Stainless Steel Cleaner',
            'brand': 'GreenClean',
            'price': 12.49,
            'rating': 4.6,
            'alternatives_count': 2
        }
        
        summary, audio_path = tts.synthesize_recommendation(product, "test_product.wav")
        print(f"✓ Summary: {summary}")
        print(f"✓ Audio: {audio_path}")
        
        print("\n" + "="*60)
        print("SUCCESS! Local TTS is working!")
        print("="*60)
        print("\nTo play audio:")
        print(f"  Linux:   ffplay {audio_path}")
        print(f"  Mac:     afplay {audio_path}")
        print(f"  Windows: start {audio_path}")
        
    except ImportError as e:
        print(f"\nERROR: {e}")
        print("\nPlease install required packages:")
        print("  pip install transformers datasets soundfile torch torchaudio")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
