import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# Set seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# 1. 예시 데이터 (사인파 + 노이즈)
data = np.sin(np.linspace(0, 50, 1000)) + np.random.normal(0, 0.1, 1000)

# 2. 정규화 (0~1 범위로 스케일링)
min_val = np.min(data)
max_val = np.max(data)
normalized_data = (data - min_val) / (max_val - min_val)

# 3. Create sequences
seq_length = 10

def create_dataset(data, seq_len):
    x, y = [], []
    for i in range(len(data) - seq_len):
        x.append(data[i:i+seq_len]) # 입력 시퀀스
        y.append(data[i+seq_len]) # 정답 값(타겟)
    return torch.tensor(x).float(), torch.tensor(y).float()

x, y = create_dataset(normalized_data, seq_length)
x = x.unsqueeze(-1)  # Add input dimension for LSTM (batch, seq, input)

# 4. Define LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1) #마지막 시점의 hidden state -> 예측값으로 반환

    def forward(self, x):
        out, _ = self.lstm(x) # LSTM 출력 (batch, seq, hidden)
        out = self.fc(out[:, -1, :]) # 마지막 시점의 hidden state 사용
        return out.squeeze()

# 5. Initialize model
model = LSTMModel()
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 6. Train the model (small number of epochs for preview)
losses = []
for epoch in range(30):
    model.train()
    pred = model(x)
    loss = loss_fn(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

# Plot training loss
plt.plot(losses)
plt.title("Training Loss (LSTM)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.tight_layout()

plt.show()
