from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import pandas as pd
from datetime import datetime

class Soil(Dataset):
    def __init__(self, is_train):
        self.is_train = is_train
        file = "data/data.csv"
        content = pd.read_csv(file)
        nrows = len(content)

        self.timestamp = list()
        self.moisture = list()
        self.temperature = list()

        self.lines = torch.zeros((nrows, 125))
        for index, row in content.iterrows():
            stamp = row[1]
            datetime_object = datetime.strptime(stamp, '%Y-%m-%d %H:%M:%S')
            timestamp = datetime_object.timestamp()
            moisture = row[2]
            temperature = row[3]
            line = row[4:]
            self.lines[index] = torch.tensor(line)
            self.timestamp.append(timestamp)
            self.moisture.append(moisture)
            self.temperature.append(temperature)

        self.start_index = 0
        self.end_index = int(int(nrows/10)*9)
        if not self.is_train:
            self.start_index = self.end_index + 1
            self.end_index = nrows - 1
        self.data_size = self.end_index - self.start_index + 1

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        index = self.start_index + idx
        return self.lines[index], self.timestamp[index], self.moisture[index], self.temperature[index]


if __name__ == "__main__":
    cid = Soil(is_train=True)
    dataloader = DataLoader(cid, batch_size=50, shuffle=True)

    for line, timestamp, moisture, temperature  in dataloader:
        print(line)
        print(timestamp)
        print(moisture)
        print(temperature)
        exit(0)
