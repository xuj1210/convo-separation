import os
from pyannote.audio import Pipeline
from dotenv import load_dotenv

# Load the Hugging Face access token from the .env file
load_dotenv()
HUGGING_FACE_ACCESS_TOKEN = os.getenv("HUGGING_FACE_ACCESS_TOKEN")

if not HUGGING_FACE_ACCESS_TOKEN:
    raise ValueError("HUGGING_FACE_ACCESS_TOKEN not found in .env file.")

def diarize_audio(audio_file_path):
    """
    Applies speaker diarization to an audio file using pyannote.audio.

    Args:
        audio_file_path (str): The path to the audio file.

    Returns:
        pyannote.core.annotation.Annotation: The diarization result.
    """
    try:
        print("Initializing pyannote.audio pipeline...")
        # Pyannote requires a Hugging Face access token to download the model
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=HUGGING_FACE_ACCESS_TOKEN
        )

        print("Applying diarization...")
        diarization = pipeline(audio_file_path)
        return diarization
    except Exception as e:
        print(f"An error occurred during diarization: {e}")
        return None

if __name__ == "__main__":
    sample_file = "data/sample.wav"  # Replace with a path to your audio file
    diarization_result = diarize_audio(sample_file)
    if diarization_result:
        print("\n--- Diarization Result ---")
        for turn, _, speaker in diarization_result.itertracks(yield_label=True):
            print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
