import os
import torch
import numpy as np
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from torch.nn.attention import SDPBackend, sdpa_kernel
from tqdm import tqdm
from pydub import AudioSegment
from contextlib import nullcontext

class WhisperTranscriber:
    """
    Audio transcription using the Whisper model pipeline
    """

    def __init__(self, model_id="openai/whisper-large-v3-turbo", use_compile=False):
        """
        Args:
            model_id (str): model ID from Hugging Face Hub
            use_compile (bool): use torch.compile
        """

        self.model_id = model_id
        self.is_compiled = use_compile

        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True
        ).to(torch.device(self.device))

        if use_compile and self.device == 'cuda':
            torch.set_float32_matmul_precision("high")
            self.model.generation_config.cache_implementation = "static"
            self.model.generation_config.max_new_tokens = 256
            self.model.forward = torch.compile(
                self.model.forward, 
                mode="reduce-overhead", 
                fullgraph=True
            )

        self.processor = AutoProcessor.from_pretrained(model_id)

        self.sampling_rate = self.processor.feature_extractor.sampling_rate

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=self.torch_dtype,
            device=self.device
        )


    def transcribe_audio(self, audio_input, warmup_passes=0):
        """
        Transcribe audio input

        Args:
            audio_file_path (str or pydub.AudioSegment or np.ndarray): The path to the audio file or an audio segment object. If passed
            an np.ndarray it must have a sampling rate of 16k
            warmup_passes (int): Number of warmup passes for torch.compile

        Returns:
            dict: The transcription result with timestamps.
        """

        
        

        if isinstance(audio_input, str):
            if not os.path.exists(audio_input):
                print(f"Error: Audio file not found at {audio_input}")
                return None
            
            prepped_input = audio_input
        elif isinstance(audio_input, AudioSegment):
            samples = np.array(audio_input.normalize().set_frame_rate(self.sampling_rate).get_array_of_samples())
            scaled_samples = samples.astype(np.float32) / audio_input.max_possible_amplitude

            prepped_input = {'sampling_rate': self.sampling_rate, 'array': scaled_samples}
            
        # Case 2: Input is a pre-scaled numpy array
        elif isinstance(audio_input, np.ndarray):
            # trust that sampling rate given is correct 16k
            prepped_input = {'sampling_rate': self.sampling_rate, 'array': audio_input}
        
        else:
            print(f"Error: Unsupported audio input type: {type(audio_input)}")
            return None

        try:
            if warmup_passes > 0:
                # 2 warmup steps
                for _ in tqdm(range(warmup_passes), desc="Warm-up step"):
                    with sdpa_kernel(SDPBackend.MATH):
                        result = self.pipe(prepped_input, generate_kwargs={"min_new_tokens": 256, "max_new_tokens": 256})

            kernel_context = sdpa_kernel(SDPBackend.MATH) if self.is_compiled else nullcontext()

            with kernel_context:
                result = self.pipe(
                    prepped_input, 
                    generate_kwargs={
                        "language": "english",
                        "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
                        "num_beams": 5,
                        "logprob_threshold": -1.0,
                        "condition_on_prev_tokens": True,
                        "return_timestamps": False
                    }
                )
            
            return result

        except Exception as e:
            print(f"An error occurred during transcription: {e}")
            return None
 

if __name__ == "__main__":
    sample_file = "data/sample.wav"
    tr = WhisperTranscriber()
    transcription_result = tr.transcribe_audio(sample_file)
    if transcription_result:
        os.makedirs("output", exist_ok=True)
        output_path = "output/transribe_result.txt"

        print(transcription_result)

        with open(output_path, "w") as f:
            chunks = transcription_result['chunks']

            for chunk in chunks:
                f.write(f'start={chunk['timestamp'][0]} stop={chunk['timestamp'][1]} text={chunk['text'][1:]}\n')


