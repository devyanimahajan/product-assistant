"""
Text-to-Speech module for the product assistant.
Supports multiple TTS providers with fragment-based synthesis.
"""

import os
from pathlib import Path
from typing import Optional, Literal
import time
from abc import ABC, abstractmethod

# Provider-specific imports (install as needed)
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    import azure.cognitiveservices.speech as speechsdk
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False


class TTSProvider(ABC):
    """Abstract base class for TTS providers."""
    
    @abstractmethod
    def synthesize(self, text: str, output_path: str) -> str:
        """
        Synthesize text to speech and save to file.
        
        Args:
            text: Text to synthesize
            output_path: Path to save audio file
            
        Returns:
            Path to saved audio file
        """
        pass


class OpenAITTS(TTSProvider):
    """OpenAI TTS provider."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "tts-1",
        voice: str = "alloy"
    ):
        """
        Initialize OpenAI TTS.
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: Model to use (tts-1 or tts-1-hd)
            voice: Voice to use (alloy, echo, fable, onyx, nova, shimmer)
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package not installed. Run: pip install openai")
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.voice = voice
    
    def synthesize(self, text: str, output_path: str) -> str:
        """Synthesize speech using OpenAI TTS."""
        print(f"[OpenAI TTS] Synthesizing {len(text)} characters...")
        
        response = self.client.audio.speech.create(
            model=self.model,
            voice=self.voice,
            input=text
        )
        
        # Save to file
        response.stream_to_file(output_path)
        
        print(f"[OpenAI TTS] Saved to {output_path}")
        return output_path


class ElevenLabsTTS(TTSProvider):
    """ElevenLabs TTS provider."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        voice_id: str = "21m00Tcm4TlvDq8ikWAM",  # Rachel voice
        model_id: str = "eleven_monolingual_v1"
    ):
        """
        Initialize ElevenLabs TTS.
        
        Args:
            api_key: ElevenLabs API key (defaults to ELEVEN_API_KEY env var)
            voice_id: Voice ID to use
            model_id: Model ID to use
        """
        if not REQUESTS_AVAILABLE:
            raise ImportError("Requests package not installed. Run: pip install requests")
        
        self.api_key = api_key or os.getenv("ELEVEN_API_KEY")
        if not self.api_key:
            raise ValueError("ElevenLabs API key not found")
        
        self.voice_id = voice_id
        self.model_id = model_id
        self.base_url = "https://api.elevenlabs.io/v1"
    
    def synthesize(self, text: str, output_path: str) -> str:
        """Synthesize speech using ElevenLabs."""
        print(f"[ElevenLabs TTS] Synthesizing {len(text)} characters...")
        
        url = f"{self.base_url}/text-to-speech/{self.voice_id}"
        
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": self.api_key
        }
        
        data = {
            "text": text,
            "model_id": self.model_id,
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.5
            }
        }
        
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()
        
        # Save to file
        with open(output_path, 'wb') as f:
            f.write(response.content)
        
        print(f"[ElevenLabs TTS] Saved to {output_path}")
        return output_path


class AzureTTS(TTSProvider):
    """Azure Speech Services TTS provider."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        region: Optional[str] = None,
        voice_name: str = "en-US-JennyNeural"
    ):
        """
        Initialize Azure TTS.
        
        Args:
            api_key: Azure Speech API key (defaults to AZURE_SPEECH_KEY env var)
            region: Azure region (defaults to AZURE_SPEECH_REGION env var)
            voice_name: Voice to use
        """
        if not AZURE_AVAILABLE:
            raise ImportError(
                "Azure Speech SDK not installed. "
                "Run: pip install azure-cognitiveservices-speech"
            )
        
        self.api_key = api_key or os.getenv("AZURE_SPEECH_KEY")
        self.region = region or os.getenv("AZURE_SPEECH_REGION")
        
        if not self.api_key or not self.region:
            raise ValueError("Azure Speech key and region not found")
        
        self.voice_name = voice_name
        self.speech_config = speechsdk.SpeechConfig(
            subscription=self.api_key,
            region=self.region
        )
        self.speech_config.speech_synthesis_voice_name = voice_name
    
    def synthesize(self, text: str, output_path: str) -> str:
        """Synthesize speech using Azure."""
        print(f"[Azure TTS] Synthesizing {len(text)} characters...")
        
        audio_config = speechsdk.audio.AudioOutputConfig(filename=output_path)
        synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=self.speech_config,
            audio_config=audio_config
        )
        
        result = synthesizer.speak_text_async(text).get()
        
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            print(f"[Azure TTS] Saved to {output_path}")
            return output_path
        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation = result.cancellation_details
            raise RuntimeError(f"Azure TTS failed: {cancellation.reason}")
        
        raise RuntimeError("Azure TTS synthesis failed")


class TextToSpeech:
    """
    Main TTS interface supporting multiple providers.
    """
    
    def __init__(
        self,
        provider: Literal["openai", "elevenlabs", "azure"] = "openai",
        output_dir: str = "./audio_output",
        **provider_kwargs
    ):
        """
        Initialize TTS with specified provider.
        
        Args:
            provider: TTS provider to use
            output_dir: Directory to save audio files
            **provider_kwargs: Additional arguments for the provider
        """
        self.provider_name = provider
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize provider
        if provider == "openai":
            self.provider = OpenAITTS(**provider_kwargs)
            self.file_extension = "mp3"
        elif provider == "elevenlabs":
            self.provider = ElevenLabsTTS(**provider_kwargs)
            self.file_extension = "mp3"
        elif provider == "azure":
            self.provider = AzureTTS(**provider_kwargs)
            self.file_extension = "wav"
        else:
            raise ValueError(f"Unknown provider: {provider}")
        
        print(f"[TTS] Initialized with provider: {provider}")
    
    def generate_summary(
        self,
        product_recommendation: dict,
        max_length: int = 150
    ) -> str:
        """
        Generate a concise spoken summary from product recommendation.
        
        Args:
            product_recommendation: Dict with product info and citations
            max_length: Max characters for summary (approx 15 seconds)
            
        Returns:
            Summary text suitable for TTS
        """
        title = product_recommendation.get('title', 'Product')
        price = product_recommendation.get('price', 0)
        rating = product_recommendation.get('rating', 0)
        brand = product_recommendation.get('brand', '')
        
        # Build concise summary
        summary_parts = []
        
        # Main recommendation
        if brand:
            summary_parts.append(f"I recommend {brand} {title}")
        else:
            summary_parts.append(f"I recommend {title}")
        
        # Price
        summary_parts.append(f"priced at ${price:.2f}")
        
        # Rating if available
        if rating > 0:
            summary_parts.append(f"with a {rating:.1f} star rating")
        
        summary = ", ".join(summary_parts) + "."
        
        # Add alternative options mention if multiple results
        alternatives_count = product_recommendation.get('alternatives_count', 0)
        if alternatives_count > 0:
            summary += f" I found {alternatives_count} similar options. Check your screen for details."
        
        # Truncate if too long
        if len(summary) > max_length:
            summary = summary[:max_length-3] + "..."
        
        return summary
    
    def synthesize_text(
        self,
        text: str,
        filename: Optional[str] = None
    ) -> str:
        """
        Synthesize text to speech and save to file.
        
        Args:
            text: Text to synthesize
            filename: Optional filename (auto-generated if not provided)
            
        Returns:
            Path to generated audio file
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        # Generate filename if not provided
        if filename is None:
            timestamp = int(time.time())
            filename = f"tts_{timestamp}.{self.file_extension}"
        
        # Ensure proper extension
        if not filename.endswith(f".{self.file_extension}"):
            filename = f"{filename}.{self.file_extension}"
        
        output_path = str(self.output_dir / filename)
        
        # Synthesize
        return self.provider.synthesize(text, output_path)
    
    def synthesize_recommendation(
        self,
        product_recommendation: dict,
        filename: Optional[str] = None
    ) -> tuple[str, str]:
        """
        Generate and synthesize speech for a product recommendation.
        
        Args:
            product_recommendation: Product recommendation dict
            filename: Optional output filename
            
        Returns:
            Tuple of (summary_text, audio_file_path)
        """
        summary = self.generate_summary(product_recommendation)
        audio_path = self.synthesize_text(summary, filename)
        return summary, audio_path


def get_tts_engine(provider: str = None, **kwargs) -> TextToSpeech:
    """
    Factory function to get TTS engine.
    
    Args:
        provider: Provider name (defaults to TTS_PROVIDER env var or "openai")
        **kwargs: Additional provider arguments
        
    Returns:
        Configured TextToSpeech instance
    """
    provider = provider or os.getenv("TTS_PROVIDER", "openai")
    return TextToSpeech(provider=provider, **kwargs)


# Example usage
if __name__ == "__main__":
    import sys
    
    # Test with OpenAI (requires OPENAI_API_KEY)
    print("\n=== TESTING TTS ===\n")
    
    try:
        # Initialize TTS
        tts = TextToSpeech(
            provider="openai",  # Change to "elevenlabs" or "azure" as needed
            output_dir="./test_audio"
        )
        
        # Example product recommendation
        product = {
            'title': 'Eco-Friendly Stainless Steel Cleaner',
            'brand': 'GreenClean',
            'price': 12.49,
            'rating': 4.6,
            'alternatives_count': 2
        }
        
        # Generate and synthesize
        print("Generating summary and audio...")
        summary, audio_path = tts.synthesize_recommendation(product)
        
        print(f"\nSummary text: {summary}")
        print(f"Audio saved to: {audio_path}")
        print(f"\nPlay with: ffplay {audio_path}")
        
    except ImportError as e:
        print(f"ERROR: {e}")
        print("\nInstall required packages:")
        print("  OpenAI: pip install openai")
        print("  ElevenLabs: pip install requests")
        print("  Azure: pip install azure-cognitiveservices-speech")
    except ValueError as e:
        print(f"ERROR: {e}")
        print("\nSet required environment variables:")
        print("  OpenAI: OPENAI_API_KEY")
        print("  ElevenLabs: ELEVEN_API_KEY")
        print("  Azure: AZURE_SPEECH_KEY, AZURE_SPEECH_REGION")
