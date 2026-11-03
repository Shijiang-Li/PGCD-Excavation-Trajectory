import torch
from torch.utils.data import Dataset
import numpy as np


class TrajectoryDataset(Dataset):
    def __init__(self, data_file):
        self.data = np.load(data_file, allow_pickle=True)  # 加载数据
        self.inputs = self.data[:, :, [1, 2, 3, 4, 12, 16, 17]]
        self.outputs = self.data[:, 1, 38].reshape(-1, 1) * 1000

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_seq = torch.tensor(self.inputs[idx], dtype=torch.float32)
        output_value = torch.tensor(self.outputs[idx], dtype=torch.float32)
        return input_seq, output_value
