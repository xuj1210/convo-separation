import os
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from torch.nn.attention import SDPBackend, sdpa_kernel
from tqdm import tqdm

if torch.backends.mps.is_available():
    device = "mps"
    print("Using Apple Silicon GPU")
elif torch.cuda.is_available():
    device = "cuda"
    print("Using NVIDIA GPU")
else:
    device = "cpu"
    print("No GPU found, using CPU")

# for torch.compile
# torch.set_float32_matmul_precision("high")

torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

def transcribe_audio(audio_file_path):
    """
    Transcribes an audio file using the Whisper model.

    Args:
        audio_file_path (str): The path to the audio file.

    Returns:
        dict: The transcription result with timestamps.
    """
    try:
        model_id = "openai/whisper-large-v3-turbo"

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True
        ).to(device)

        # Enable static cache and compile the forward pass
        # model.generation_config.cache_implementation = "static"
        # model.generation_config.max_new_tokens = 256
        # model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)


        processor = AutoProcessor.from_pretrained(model_id)

        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=device
        )

        result = pipe(
            audio_file_path, 
            return_timestamps=True, 
            generate_kwargs={"language": "english"}
        )

        # # 2 warmup steps
        # for _ in tqdm(range(2), desc="Warm-up step"):
        #     with sdpa_kernel(SDPBackend.MATH):
        #         result = pipe(audio_file_path, generate_kwargs={"min_new_tokens": 256, "max_new_tokens": 256})

        # with sdpa_kernel(SDPBackend.MATH):
        #     result = pipe(audio_file_path)
        
        return result
    except Exception as e:
        print(f"An error occurred during transcription: {e}")
        return None

if __name__ == "__main__":
    sample_file = "data/sample.wav"
    transcription_result = transcribe_audio(sample_file)
    if transcription_result:
        os.makedirs("output", exist_ok=True)
        output_path = "output/transribe_result.txt"

        print(f"Writing transcription result to {output_path}")

        with open(output_path, "w") as f:
            chunks = transcription_result['chunks']

            for chunk in chunks:
                f.write(f'start={chunk['timestamp'][0]} stop={chunk['timestamp'][1]} text={chunk['text'][1:]}\n')


