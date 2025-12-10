"""
Record Audio from Microphone

Simple script to record audio for testing ASR.
"""

import sys

try:
    import sounddevice as sd
    import numpy as np
    import wave
except ImportError:
    print("\n❌ Error: sounddevice not installed")
    print("\nInstall with:")
    print("   pip install sounddevice numpy")
    print()
    sys.exit(1)


def record_audio(output_file: str = "recording.wav", duration: int = 5):
    """
    Record audio from microphone.
    
    Args:
        output_file: Output filename
        duration: Recording duration in seconds
    """
    print("\n" + "="*60)
    print("MICROPHONE RECORDING")
    print("="*60)
    print(f"\nOutput file: {output_file}")
    print(f"Duration: {duration} seconds")
    print(f"Sample rate: 16000 Hz")
    
    # Countdown
    print("\n🎤 Recording will start in:")
    import time
    for i in range(3, 0, -1):
        print(f"   {i}...")
        time.sleep(1)
    
    print("\n🔴 RECORDING NOW - Speak clearly into your microphone!")
    
    # Record
    sample_rate = 16000
    audio_data = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype=np.int16
    )
    
    # Wait for recording to finish
    sd.wait()
    
    print("✅ Recording complete!")
    
    # Save to file
    print(f"\nSaving to: {output_file}")
    
    with wave.open(output_file, 'wb') as wf:
        wf.setnchannels(1)  # Mono
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())
    
    print("✅ Saved successfully!")
    
    # Ask if user wants to transcribe
    print("\n" + "="*60)
    answer = input("\nTranscribe this recording now? (y/n) [y]: ").strip().lower()
    
    if answer != 'n':
        transcribe_recording(output_file)


def transcribe_recording(audio_file: str):
    """Transcribe the recording"""
    try:
        from asr_tool import transcribe_audio
    except ImportError:
        print("\n❌ Error: Cannot import asr_tool")
        print("   Make sure asr_tool.py is in the same directory")
        return
    
    print("\n" + "="*60)
    print("TRANSCRIPTION")
    print("="*60)
    print(f"\nFile: {audio_file}")
    print("Model: base")
    print("\nTranscribing...\n")
    
    result = transcribe_audio(
        audio_path=audio_file,
        model_size="base",
        return_timestamps=True
    )
    
    print("="*60)
    
    if result["success"]:
        print("✅ SUCCESS\n")
        print(f"📝 You said:")
        print(f'   "{result["text"]}"')
        print(f"\n🌍 Language: {result['language'].upper()}")
        
        if "segments" in result:
            print(f"\n📊 Segments:\n")
            for seg in result["segments"]:
                print(f"   [{seg['start']:6.2f}s - {seg['end']:6.2f}s] {seg['text']}")
    else:
        print("❌ FAILED\n")
        print(f"Error: {result.get('error')}")
    
    print("\n" + "="*60)


def main():
    """Main function"""
    print("\n" + "="*60)
    print("ASR - RECORD FROM MICROPHONE")
    print("="*60)
    
    # Get parameters
    if len(sys.argv) > 1:
        output_file = sys.argv[1]
    else:
        output_file = input("\nOutput filename [recording.wav]: ").strip()
        if not output_file:
            output_file = "recording.wav"
    
    if len(sys.argv) > 2:
        duration = int(sys.argv[2])
    else:
        duration_input = input("Recording duration in seconds [5]: ").strip()
        duration = int(duration_input) if duration_input else 5
    
    # Record
    try:
        record_audio(output_file, duration)
    except KeyboardInterrupt:
        print("\n\n❌ Recording cancelled")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    
    print()


if __name__ == "__main__":
    main()
