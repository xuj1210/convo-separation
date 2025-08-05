from diarize import Diarizer
import pandas as pd


df = pd.read_csv('output/diarization_result.csv')


OLLAMA_MODEL='cogito:14b'

diarizer = Diarizer(ollama_model=OLLAMA_MODEL)

cleaned = diarizer._clean_with_ollama(df)
cleaned.to_csv('output/cleaned_result.csv')