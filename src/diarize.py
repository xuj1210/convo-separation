import os
import torch
import requests
import pandas as pd
import numpy as np
from pyannote.audio import Pipeline
from dotenv import load_dotenv
from transcribe import WhisperTranscriber
from pydub import AudioSegment
from tqdm import tqdm
from google import genai
from audio_preprocess import preprocess_audio

load_dotenv()
HUGGING_FACE_ACCESS_TOKEN = os.getenv("HUGGING_FACE_ACCESS_TOKEN")
if not HUGGING_FACE_ACCESS_TOKEN:
    raise ValueError("HUGGING_FACE_ACCESS_TOKEN not found in .env file.")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file.")

class Diarizer:
    """
    Speech diarization using pyannote.audio and transcription using Whisper
    """

    def __init__(self, model_id="pyannote/speaker-diarization-3.1", 
                 use_compile=False):
        """
        Args:
            model_id (str): model ID
        """

        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        
        self.pipe = Pipeline.from_pretrained(
            model_id,
            use_auth_token=HUGGING_FACE_ACCESS_TOKEN
        ).to(torch.device(self.device))

        self.transcriber = WhisperTranscriber(use_compile=use_compile)
        
        self.ollama_api_url = "http://localhost:11434/api/chat"

        self.core_system_prompt = (
            "You are an expert conversation editor. Your task is to clean up a raw conversation transcript. "
            "You must read and understand the entire conversation to infer meaning and make accurate corrections. "
            "Your output MUST follow these specific instructions for each line:\n\n"
            "1. **Use Context for Corrections:** Use context clues from surrounding dialogue to ensure the conversation makes sense in all areas. For example, you might have to correct words "
            "or sentences that makes no sense in the context of the overall dialogue.\n"
            "2. **Correct Errors:** The transcript was automatically generated from audio. Correct grammar, spelling, and punctuation errors. Also fix words or phrases that are likely incorrect because they sound similar to the correct word when spoken (e.g., 'know' instead of 'no').\n"            
            # "3. **Maintain Originality:** Do not add or remove any information. Preserve the original meaning and intent of the speakers exactly as it was. Do not combine or split sentences.\n"
            "3. **Preserve Structure:** The number of lines in your output must be identical to the number of lines in the input. Each line should begin with the speaker label, followed by the corrected dialogue.\n\n"
            "Your final output MUST be in this exact format, do not include any additional text besides the conversation:\n"
            "SPEAKER_[number]: [Corrected text]\n"
            "SPEAKER_[number]: [Corrected text]\n"
        )

        self.clean_method = 'gemini'


    def diarize_audio(self, audio_file_path: str):
        """
        Applies speaker diarization to an audio file

        Args:
            audio_file_path (str): Path to the audio file

        Returns:
            pandas.DataFrame: 
        """

        if not os.path.exists(audio_file_path):
            print(f"Error: Audio file not found at {audio_file_path}")
            return None
        
        try:
            target_sample_rate = self.transcriber.sampling_rate
            waveform, _ = preprocess_audio(audio_file_path, target_sample_rate=target_sample_rate)
            

        except Exception as e:
            print(f"Error loading audio file: {e}")
            return None
        
        diarization = self.pipe({
            "waveform": waveform,
            "sample_rate": target_sample_rate
        })

        results = []

        for turn, _, speaker in tqdm(diarization.itertracks(yield_label=True)):
            start_index = int((turn.start) * target_sample_rate)
            end_index = int((turn.end) * target_sample_rate)
            segment = waveform.squeeze()[start_index:end_index]
            transcription = self.transcriber.transcribe_audio(segment)
            if transcription:
                results.append({
                    'speaker': speaker,
                    'text': transcription['text'].strip(),
                    'start': round(turn.start, 3),
                    'end': round(turn.end, 3)
                })
        
        return self._clean_transcript(pd.DataFrame(results), method=self.clean_method)

    def _get_cleaned_text(self, conversation_df: pd.DataFrame):
        """
        Helper to group conversation turns for cleaning and prepares the prompt.
        """
        if conversation_df.empty:
            return conversation_df, None

        grouped_turns = []
        current_speaker = None
        current_text = ""
        current_start = None
        current_end = None

        for _, row in conversation_df.iterrows():
            end_time = row['end']
            speaker = row['speaker']
            text = row['text']

            if speaker != current_speaker and current_speaker is not None:
                grouped_turns.append({
                    'speaker': current_speaker,
                    'text': current_text.strip(),
                    'start': current_start,
                    'end': current_end,
                })
                current_text = ''
            
            current_speaker = speaker
            current_text += f' {text}'
            if current_start is None:
                current_start = row['start']
            current_end = end_time

        if current_speaker is not None:
            grouped_turns.append({
                'speaker': current_speaker,
                'text': current_text.strip(),
                'start': current_start,
                'end': current_end,
            })
        
        raw_transcript = '\n'.join([f"{turn['speaker']}: {turn['text']}" for turn in grouped_turns])
        return grouped_turns, raw_transcript

    def _apply_llm_response(self, grouped_turns, cleaned_text):
        """
        Parses LLM response of cleaning/editing conversation
        """

        cleaned = []
        cleaned_lines = cleaned_text.strip().split('\n')
           
        if len(cleaned_lines) != len(grouped_turns):
            print(f"Warning: Number of turns in LLM response ({len(cleaned_lines)}) does not match input ({len(grouped_turns)}). Falling back to raw transcription.")
            return pd.DataFrame(grouped_turns)

        for i, line in enumerate(cleaned_lines):
            turn_data = grouped_turns[i]
            speaker = turn_data['speaker']

            if not line.startswith(f'{speaker}:'):
                print(f"Warning: LLM response format mismatch. Skipping line: {line}")
                continue

            cleaned_text = line.split(':', 1)[1].strip()
            cleaned.append({
                'speaker': speaker,
                'text': cleaned_text,
                'start': turn_data['start'],
                'end': turn_data['end']
            })
        return pd.DataFrame(cleaned)

    def _clean_with_ollama(self, conversation_df: pd.DataFrame, ollama_model='gemma3n:e4b') -> pd.DataFrame:
        """
        Cleans transcription errors from raw conversation transcription using local LLM via Ollama API
        """
        grouped_turns, raw_transcript = self._get_cleaned_text(conversation_df)
        if raw_transcript is None:
            return conversation_df

        user_message = f"Edit the following conversation:\n\n{raw_transcript}"

        payload = {
            "model": ollama_model,
            "messages": [
                {"role": "system", "content": self.core_system_prompt},
                {"role": "user", "content": user_message}
            ],
            "stream": False,
            "options": {"temperature": 0.2}
        }

        try:
            print(f'Sending prompt to Ollama model: {ollama_model}')
            response = requests.post(self.ollama_api_url, json=payload, timeout=300)
            response.raise_for_status()
            print('Successfully received response from Ollama')
            cleaned_text = response.json()['message']['content']
            return self._apply_llm_response(grouped_turns, cleaned_text)
            
        except requests.exceptions.RequestException as e:
            print(f"Error communicating with the Ollama API: {e}")
            print("Falling back to the raw transcription.")
            return conversation_df

    def _clean_with_gemini(self, conversation_df: pd.DataFrame, model='gemma-3-27b-it') -> pd.DataFrame:
        """
        Cleans grammar and transcription errors using a Gemini LLM
        """
        grouped_turns, raw_transcript = self._get_cleaned_text(conversation_df)
        if raw_transcript is None:
            return conversation_df

        client = genai.Client(api_key=GEMINI_API_KEY)

        full_prompt = (
            f"{self.core_system_prompt}\n"
            "Additionally, please refrain from switching existing ’ and ' characters.\n\n"
            f"Edit the following conversation:\n\n{raw_transcript}"
        )

        try:
            print(f'Sending prompt to Gemini model: {model}')
            response = client.models.generate_content(
                model=model,
                contents=full_prompt
            )
            return self._apply_llm_response(grouped_turns, response.text)

        except requests.exceptions.RequestException as e:
            print(f"Error calling Gemini: {e}")
            print("Falling back to the raw transcription.")
            return conversation_df

    def _clean_transcript(self, conversation_df: pd.DataFrame, method: str = 'ollama', ollama_model='gemma-3-27b-it') -> pd.DataFrame:
        """
        Cleans the conversation transcript using the specified LLM method.

        Args:
            conversation_df (pd.DataFrame): The raw conversation transcript.
            method (str): The cleaning method to use ('ollama' or 'huggingface').

        Returns:
            pd.DataFrame: The cleaned transcript.
        """
        if method == 'ollama':
            return self._clean_with_ollama(conversation_df, ollama_model=ollama_model)
        elif method == 'gemini':
            return self._clean_with_gemini(conversation_df)
        else:
            print(f"Error: Unknown cleaning method '{method}'. Using raw transcription.")
            return conversation_df

    

if __name__ == "__main__":
    sample_file = "data/sample.wav"
    d = Diarizer()
    diarization_result = d.diarize_audio(sample_file)

    # test = pd.read_csv('output/grouped_result.csv')
    # d._clean_transcript(test, method='ollama', ollama_model='cogito:8b').to_csv('test_cleaned.csv')

    if diarization_result is not None and not diarization_result.empty:
        os.makedirs("output", exist_ok=True)
        output_file_path = "output/diarization_result.csv"
        print(f"Writing diarization result to {output_file_path}")
        diarization_result.to_csv(output_file_path)
