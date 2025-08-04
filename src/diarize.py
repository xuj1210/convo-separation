import os
import torch
from pyannote.audio import Pipeline
from dotenv import load_dotenv

# Load the Hugging Face access token from the .env file
load_dotenv()
HUGGING_FACE_ACCESS_TOKEN = os.getenv("HUGGING_FACE_ACCESS_TOKEN")

if not HUGGING_FACE_ACCESS_TOKEN:
    raise ValueError("HUGGING_FACE_ACCESS_TOKEN not found in .env file.")

if torch.backends.mps.is_available():
    device = "mps"
    print("Using Apple Silicon GPU")
elif torch.cuda.is_available():
    device = "cuda"
    print("Using NVIDIA GPU")
else:
    device = "cpu"
    print("No GPU found, using CPU")


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
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=HUGGING_FACE_ACCESS_TOKEN
        )

        pipeline.to(torch.device(device))

        print("Applying diarization...")
        diarization = pipeline(audio_file_path)
        return diarization
    
    except Exception as e:
        print(f"An error occurred during diarization: {e}")
        return None

if __name__ == "__main__":
    sample_file = "data/sample.wav"
    diarization_result = diarize_audio(sample_file)
    if diarization_result:
        os.makedirs("output", exist_ok=True)
        output_file_path = "output/diarization_result.txt"
        print(f"\n--- Diarization Result ---")
        print(f"Writing diarization result to {output_file_path}")

        # writing result
        with open(output_file_path, "w") as f:
            for turn, _, speaker in diarization_result.itertracks(yield_label=True):
                f.write(f"start={turn.start:.2f} stop={turn.end:.2f} {speaker}\n")