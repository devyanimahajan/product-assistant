"""
Quick ASR Test - Simplest possible test

Just run: python quick_test_asr.py your_audio.wav
"""

import sys
import os

def quick_test():
    """Simplest test possible"""
    
    # Check arguments
    if len(sys.argv) < 2:
        print("\n📖 Usage:")
        print("   python quick_test_asr.py <audio_file>")
        print("\nExample:")
        print("   python quick_test_asr.py sample.wav")
        print("   python quick_test_asr.py recording.mp3")
        print()
        return
    
    audio_path = sys.argv[1]
    
    # Check file exists
    if not os.path.exists(audio_path):
        print(f"\n❌ Error: File not found: {audio_path}")
        return
    
    print("\n" + "="*60)
    print("ASR QUICK TEST")
    print("="*60)
    print(f"\nFile: {audio_path}")
    
    # Import ASR
    try:
        from asr_tool import transcribe_audio
    except ImportError:
        print("\n❌ Error: Cannot import asr_tool")
        print("   Make sure asr_tool.py is in the same directory")
        return
    
    print("Model: base")
    print("\nTranscribing... (this may take a few seconds)\n")
    
    # Transcribe
    result = transcribe_audio(
        audio_path=audio_path,
        model_size="base",
        return_timestamps=True
    )
    
    # Show results
    print("="*60)
    
    if result["success"]:
        print("✅ SUCCESS\n")
        print(f"📝 Transcript:")
        print(f'   "{result["text"]}"')
        print(f"\n🌍 Language: {result['language'].upper()}")
        
        if "duration" in result:
            print(f"⏱️  Duration: {result['duration']:.1f} seconds")
        
        if "segments" in result and len(result["segments"]) > 0:
            print(f"\n📊 Timestamped Segments:\n")
            for seg in result["segments"]:
                print(f"   [{seg['start']:6.2f}s - {seg['end']:6.2f}s] {seg['text']}")
        
        print("\n" + "="*60)
        print("✅ Test completed successfully!")
    else:
        print("❌ FAILED\n")
        print(f"Error: {result.get('error', 'Unknown error')}")
        print("\n" + "="*60)
    
    print()


if __name__ == "__main__":
    quick_test()
