import pandas as pd

df = pd.read_csv("output/diarization_result.csv")

print(df.loc[0, 'segment'])