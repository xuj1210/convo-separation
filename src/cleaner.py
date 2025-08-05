from diarize import Diarizer
import pandas as pd


df = pd.read_csv('output/diarization_result.csv')


OLLAMA_MODEL='cogito:14b'

diarizer = Diarizer()

cleaned = diarizer.clean_transcript(df, method='gemini')
cleaned.to_csv('output/cleaned_result.csv')