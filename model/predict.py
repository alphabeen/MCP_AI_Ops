import torch
import numpy as np
from .model import LSTMModel

# 모델 불러오기
model = LSTMModel()
model.load_state_dict(torch.load("app/lstm_model.pth"))
model.eval()

def predict_lstm(input_seq: list[float]) -> float:
    seq = np.array(input_seq).reshape(1, -1, 1)  # (batch, seq_len, feature)
    tensor_seq = torch.FloatTensor(seq)

    with torch.no_grad():
        output = model(tensor_seq)
    return output.item()
