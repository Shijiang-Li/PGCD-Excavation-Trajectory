import torch
from torch.utils.data import Dataset
import numpy as np


class TrajectoryDataset(Dataset):
    def __init__(self, data_file):
        """
        初始化，加载 .npy 文件并解析为变长序列。
        """
        self.data = np.load(data_file, allow_pickle=True)  # 加载数据
        self.inputs = [torch.tensor(sample[:, [1, 2, 3, 4, 12, 13, 14, 15, 16, 17]], dtype=torch.float32)
                       for sample in self.data]
        self.targets = [torch.tensor(sample[:, [46, 47, 48]], dtype=torch.float32) for sample in self.data]

    def __len__(self):
        """
        数据集的样本数量
        """
        return len(self.inputs)

    def __getitem__(self, idx):
        """
        返回一个样本的输入和目标
        """
        return self.inputs[idx], self.targets[idx]
