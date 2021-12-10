from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
import time
import soil_dataset
from machine import Machine

def train():
    NUM_EPOCHS = 3
    soil = soil_dataset.Soil(is_train=True)
    dataloader = DataLoader(soil, batch_size=1, shuffle=False)
    model = Machine()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss = None
    print("Training started...")
    start_time = time.time()
    for epoch  in range(0, NUM_EPOCHS):
        for line, timestamp, moisture, temperature in dataloader:
            pred = model(line)
            print(pred)
            exit(0)
            #loss = F.nll_loss(y_pred, y_true)
            #loss.backward()
            optimizer.step()
        print(f'Epoch:{epoch + 1}, Loss:{loss.item():.4f}')
    end_time = time.time()
    print(f"Training end. Time required: {round(end_time-start_time,2)}")
    return model


if __name__ == "__main__":
    train()

