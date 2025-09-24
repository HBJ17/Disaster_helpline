# modules/speech_to_text.py
import librosa
from modules.model_loader import whisper_model

# Whisper was trained on 16kHz audio.
WHISPER_SAMPLE_RATE = 16000

def transcribe_audio(audio_path):
    """
    Transcribes an audio file using the pre-loaded Whisper model.

    This version loads the audio into memory first to be more robust against
    file path issues, especially on Windows.

    Args:
        audio_path (str): The path to the audio file.

    Returns:
        str: The transcribed text, or an empty string if transcription fails.
    """
    if whisper_model is None:
        print(" ⚠️ Whisper model not available. Skipping transcription.")
        return ""

    try:
        # Load the audio file using librosa.
        # This function automatically resamples to the target sample rate (16kHz)
        # and converts the audio to a mono, float32 NumPy array, which is
        # exactly what Whisper's transcribe method expects.
        waveform, _ = librosa.load(audio_path, sr=WHISPER_SAMPLE_RATE, mono=True)

        # Transcribe the in-memory audio waveform.
        # Set fp16=False if you are running on a CPU.
        result = whisper_model.transcribe(waveform, fp16=False)

        transcript = result.get("text", "").strip()
        return transcript

    except Exception as e:
        # Don't crash the main pipeline; upstream handles empty transcripts.
        print(f" ⚠️ Error in Whisper transcription: {e}")
        return ""