import os
import torch
import numpy as np
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from torch.nn.attention import SDPBackend, sdpa_kernel
from tqdm import tqdm
from pydub import AudioSegment
from contextlib import nullcontext
from audio_preprocess import preprocess_audio

class WhisperTranscriber:
    """
    Audio transcription using the Whisper model pipeline
    """

    def __init__(self, model_id="openai/whisper-large-v3-turbo"):
        """
        Args:
            model_id (str): model ID from Hugging Face Hub
            use_compile (bool): use torch.compile
        """

        self.model_id = model_id

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        self.model = WhisperForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True
        ).to(self.device)
        self.processor = WhisperProcessor.from_pretrained(model_id)

        self.sampling_rate = self.processor.feature_extractor.sampling_rate
  
    def transcribe_diarized_audio(self, waveform: torch.Tensor, diarization):
        """
        Transcribes diarized audio segments with context from previous segments.
        
        Args:
            waveform (torch.Tensor): The audio waveform.
            diarization (pyannote.Core.Annotation): Diarization information.
        
        Returns:
            list: A list of dictionaries with transcription, speaker, and timestamps.
        """
        waveform = waveform.squeeze()

        transcriptions = []
        previous_text = ""

        # context_history = [] 
        # max_context_length = 3

        for segment, _, speaker in tqdm(diarization.itertracks(yield_label=True), desc="Transcribing segments"):
            start_time = int(segment.start * self.sampling_rate)
            end_time = int(segment.end * self.sampling_rate)
            
            audio_chunk = waveform[start_time:end_time]
            
            input_features = self.processor(
                audio_chunk.cpu(), 
                sampling_rate=self.sampling_rate, 
                return_tensors="pt"
            ).input_features.to(self.device, self.torch_dtype)

            prompt_ids = self.processor.get_prompt_ids(
                previous_text,
                return_tensors="pt"
            ).to(self.device)

            generated_ids = self.model.generate(
                input_features=input_features,
                # condition_on_prev_tokens=True,
                # temperature=0.2,
                num_beams=4,
                num_beam_groups=2,
                diversity_penalty=1.0,
                prompt_ids=prompt_ids,
                language='english',
                task='transcribe',
                temperature=0.0,
                attention_mask=torch.ones_like(input_features, device=self.device)
            )

            transcription = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )[0]
            
            print(transcription)
            previous_text = transcription
            
            transcriptions.append({
                "speaker": speaker,
                "start": round(segment.start, 2),
                "end": round(segment.end, 2),
                "text": transcription
            })
            
        return transcriptions

        
    # def transcribe_file(self, audio_file_path: str):
    #     if not os.path.exists(audio_file_path):
    #         print(f"Error: Audio file not found at {audio_file_path}")
    #         return None
        
    #     waveform, _ = preprocess_audio(audio_file_path, target_sample_rate=self.sampling_rate)
    #     waveform = waveform.squeeze().to(self.device)
        
    #     transcription, _ = self._transcribe_segment(waveform)
        
    #     return {'text': transcription}
    
 

# if __name__ == "__main__":
#     sample_file = "data/sample.wav"
#     tr = WhisperTranscriber()
#     transcription_result = tr.transcribe_audio(sample_file)
#     if transcription_result:
#         os.makedirs("output", exist_ok=True)
#         output_path = "output/transribe_result.txt"

#         print(transcription_result)

#         with open(output_path, "w") as f:
#             chunks = transcription_result['chunks']

#             for chunk in chunks:
#                 f.write(f'start={chunk['timestamp'][0]} stop={chunk['timestamp'][1]} text={chunk['text'][1:]}\n')


