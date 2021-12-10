import torch
import pandas as pd
from datetime import datetime
import time

file = "data/data.csv"
content = pd.read_csv(file)
nrows = len(content)
lines = torch.zeros((nrows, 125))
print()
for index, row in content.iterrows():
    stamp = row[1]
    datetime_object = datetime.strptime(stamp, '%Y-%m-%d %H:%M:%S')
    timestamp = datetime_object.timestamp()
    moisture = row[2]
    temperature = row[3]
    line = row[4:]
    lines[index] = torch.tensor(line)

print(f'{lines[-1,-1].item():.10f}')


