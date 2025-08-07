import pandas as pd
from diarize import Diarizer
import os

d = Diarizer()

file_name = 'CAR0001'
df = pd.read_csv(f'output/{file_name}.csv')

_, transcript = d._get_cleaned_text(df)

with open(f'output/{file_name}.txt', 'w') as f:
    f.write(transcript)