"""
GRU模型
"""

try:
    import nni  # only needed for hyper-parameter search
except ImportError:
    nni = None
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch.optim as optim
from torch.nn.utils import weight_norm
from Forward_model_diff.forward_model3_multi_task.data_load import *
import matplotlib.pyplot as plt


def collate_fn(batch):
    """
    处理变长序列，填充序列到相同长度并记录实际长度
    """
    inputs, targets = zip(*batch)
    lengths = torch.tensor([seq.size(0) for seq in inputs])  # 记录每个序列的长度
    padded_inputs = pad_sequence(inputs, batch_first=True)  # 填充输入
    padded_targets = pad_sequence(targets, batch_first=True)  # 填充目标
    return padded_inputs, padded_targets, lengths


class BiGRU_Block(nn.Module):
    def __init__(self, input_size, hidden_size_gru, num_layers):
        super(BiGRU_Block, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size_gru, num_layers, batch_first=True, dropout=0, bidirectional=True)

    def forward(self, x, lengths):
        # 打包序列，忽略填充部分
        packed_input = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.gru(packed_input)
        out, _ = pad_packed_sequence(packed_output, batch_first=True)
        return out


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding,
                                           dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding,
                                           dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i  # 膨胀系数
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch_size, input_channels, seq_len)
        x = self.network(x)
        out = x.permute(0, 2, 1)  # (batch_size, seq_len, output_channels)
        return out


class MLP_Block(nn.Module):
    def __init__(self, mlp_input_dim, mlp_hidden_dims, mlp_out_dim):
        super(MLP_Block, self).__init__()
        layers = [nn.Linear(mlp_input_dim, mlp_hidden_dims[0]), nn.ReLU()]  # 输入层到第一隐层

        # 隐层到隐层的连接
        for i in range(1, len(mlp_hidden_dims)):
            layers.append(nn.Linear(mlp_hidden_dims[i - 1], mlp_hidden_dims[i]))
            layers.append(nn.ReLU())

        # 输出层
        layers.append(nn.Linear(mlp_hidden_dims[-1], mlp_out_dim))

        # 将所有层组合为一个 Sequential 模块
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class BiGRU_TCN_Multi_model(nn.Module):
    def __init__(self, input_size, hidden_size_bigru, num_layers_bigru, num_channels_tcn,
                 Ft_mlp_hidden, Fn_mlp_hidden, Mr_mlp_hidden):
        super(BiGRU_TCN_Multi_model, self).__init__()
        self.gru_model = BiGRU_Block(input_size, hidden_size_bigru, num_layers_bigru)
        self.tcn_model = TemporalConvNet(hidden_size_bigru * 2, num_channels_tcn, kernel_size=2, dropout=0)

        self.mlp_Ft = MLP_Block(num_channels_tcn[-1], Ft_mlp_hidden, 1)
        self.mlp_Fn = MLP_Block(num_channels_tcn[-1], Fn_mlp_hidden, 1)
        self.mlp_Mr = MLP_Block(num_channels_tcn[-1], Mr_mlp_hidden, 1)

    def forward(self, x, lengths):
        x = self.gru_model(x, lengths)
        x = self.tcn_model(x)

        out_Ft = self.mlp_Ft(x)
        out_Fn = self.mlp_Fn(x)
        out_Mr = self.mlp_Mr(x)
        out = torch.cat([out_Ft, out_Fn, out_Mr], dim=2)

        return out


def train_model(train_loader, model, optimizer, device, criterion, max_min_force, scheduler, lambda_loss):
    model.train()
    train_loss = 0.0
    for input_batch, targets_batch, lengths in train_loader:
        input_batch = input_batch.to(device).to(torch.float32)  # 输入特征
        targets_batch = targets_batch.to(device).to(torch.float32)  # 真实值

        optimizer.zero_grad()
        outputs = model(input_batch, lengths)
        # loss = criterion(outputs_1, targets_batch) + criterion(outputs_2, targets_batch)
        loss = (lambda_loss[0] * criterion(outputs[:, :, 0] / max_min_force[0],
                                           targets_batch[:, :, 0] / max_min_force[0]) +
                lambda_loss[1] * criterion(outputs[:, :, 1] / max_min_force[1],
                                           targets_batch[:, :, 1] / max_min_force[1]) +
                lambda_loss[2] * criterion(outputs[:, :, 2] / max_min_force[2],
                                           targets_batch[:, :, 2] / max_min_force[2]))
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    epoch_loss = train_loss / len(train_loader)
    scheduler.step()
    return epoch_loss


def test_model(test_loader, model, device, criterion, max_min_force, lambda_loss):
    model.eval()
    with torch.no_grad():
        test_loss = 0.0
        for input_batch, targets_batch, lengths in test_loader:
            input_batch = input_batch.to(device).to(torch.float32)  # 输入特征
            targets_batch = targets_batch.to(device)  # 真实值

            outputs = model(input_batch, lengths)
            loss = (lambda_loss[0] * criterion(outputs[:, :, 0] / max_min_force[0],
                                               targets_batch[:, :, 0] / max_min_force[0]) +
                    lambda_loss[1] * criterion(outputs[:, :, 1] / max_min_force[1],
                                               targets_batch[:, :, 1] / max_min_force[1]) +
                    lambda_loss[2] * criterion(outputs[:, :, 2] / max_min_force[2],
                                               targets_batch[:, :, 2] / max_min_force[2]))
            test_loss += loss.item()
        epoch_loss = test_loss / len(test_loader)
    return epoch_loss


def plot_train_test_loss(epochs_, train_losses_arr, test_losses_arr):
    plt.plot(range(epochs_), train_losses_arr, color=(40 / 255, 124 / 255, 183 / 255), linewidth=1, label='train')
    plt.plot(range(epochs_), test_losses_arr, color=(255 / 255, 156 / 255, 75 / 255), linewidth=1, label='test')
    legend = plt.legend(loc='upper right', ncol=2)
    legend.get_frame().set_alpha(0)  # 设置图例的透明度为0
    plt.show()


# 使用示例
def main():
    # device = "cpu"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    params = {'batch_size': 64, 'hidden_size_bigru': 64, 'num_layers_bigru': 1, 'hidden_dim_tcn': 64,
              'num_layers_tcn': 3,
              'Ft_mlp_hidden': 32, 'Fn_mlp_hidden': 64, 'Mr_mlp_hidden': 64, 'mlp_layers': 2,
              'learning_rate': 0.01489855719828758}

    optimized_params = nni.get_next_parameter()
    params.update(optimized_params)
    print(params)

    batch_size = params['batch_size']
    train_dataset = TrajectoryDataset(data_file="./../../OriginalData"
                                                "/All_material_data_diff"
                                                "/split_data_seq_8_2_same_check_norm"
                                                "/data_train_get.npy")

    # 找到训练集中，各个力的最大最小值
    max_value_Ft, max_value_Fn, max_value_Mr = float('-inf'), float('-inf'), float('-inf')
    min_value_Ft, min_value_Fn, min_value_Mr = float('inf'), float('inf'), float('inf')
    for idx in range(len(train_dataset)):
        input_seq, output_seq = train_dataset[idx]
        max_value_Ft = max(max_value_Ft, output_seq[:, 0].max().item())
        min_value_Ft = min(min_value_Ft, output_seq[:, 0].min().item())
        max_value_Fn = max(max_value_Fn, output_seq[:, 1].max().item())
        min_value_Fn = min(min_value_Fn, output_seq[:, 1].min().item())
        max_value_Mr = max(max_value_Mr, output_seq[:, 2].max().item())
        min_value_Mr = min(min_value_Mr, output_seq[:, 2].min().item())
    max_min_force = np.array([(max_value_Ft - min_value_Ft),
                              (max_value_Fn - min_value_Fn),
                              (max_value_Mr - min_value_Mr)]) * 0.01

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    # 加载测试数据
    test_dataset = TrajectoryDataset(data_file="./../../OriginalData"
                                               "/All_material_data_diff"
                                               "/split_data_seq_8_2_same_check_norm"
                                               "/data_test_f_get.npy")
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    test_dataset_soil = TrajectoryDataset(data_file="./../../OriginalData"
                                                    "/All_material_data_diff"
                                                    "/split_data_seq_8_2_same_check_norm"
                                                    "/data_test_uf_get.npy")
    test_loader_soil = DataLoader(test_dataset_soil, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # 定义模型、损失函数和优化器
    hidden_size_list = []
    for i in range(params['num_layers_tcn']):
        hidden_size_list.append(params['hidden_dim_tcn'])
    Ft_mlp_hidden_list = []
    Fn_mlp_hidden_list = []
    Mr_mlp_hidden_list = []
    for i in range(params['mlp_layers']):
        Ft_mlp_hidden_list.append(params['Ft_mlp_hidden'])
        Fn_mlp_hidden_list.append(params['Fn_mlp_hidden'])
        Mr_mlp_hidden_list.append(params['Mr_mlp_hidden'])
    model = BiGRU_TCN_Multi_model(input_size=10, hidden_size_bigru=params['hidden_size_bigru'],
                                  num_layers_bigru=params['num_layers_bigru'], num_channels_tcn=hidden_size_list,
                                  Ft_mlp_hidden=Ft_mlp_hidden_list, Fn_mlp_hidden=Fn_mlp_hidden_list,
                                  Mr_mlp_hidden=Mr_mlp_hidden_list).to(device)
    lambda_loss = np.array([0.3, 0.3, 0.3])

    criterion = nn.MSELoss()
    # criterion = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=params['learning_rate'])

    # 训练模型
    epochs = 500
    train_losses_ = np.zeros(epochs)
    # test_losses = 0
    test_losses_ = np.zeros(epochs)

    best_test = float('inf')
    best_test_soil = float('inf')
    best_validation_loss = float('inf')
    early_stop_patience = 300  # 五轮不下降触发
    early_stop_counter = 0
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=epochs // 3, gamma=0.5)
    for t in range(epochs):
        print(f"-------------------------------\nEpoch {t + 1}")
        train_losses = train_model(train_loader, model, optimizer, device, criterion, max_min_force, scheduler,
                                   lambda_loss)
        test_losses = test_model(test_loader, model, device, criterion, max_min_force, lambda_loss)
        test_losses_soil = test_model(test_loader_soil, model, device, criterion, max_min_force, lambda_loss)
        train_losses_[t] = train_losses
        test_losses_[t] = test_losses
        nni.report_intermediate_result(train_losses)
        nni.report_intermediate_result(test_losses)
        nni.report_intermediate_result(test_losses_soil)

        k = 1.4
        if (test_losses + k * test_losses_soil) < best_validation_loss:
            best_test = test_losses
            best_test_soil = test_losses_soil
            best_validation_loss = (test_losses + k * test_losses_soil)
            early_stop_counter = 0
            torch.save(model, f"./forward_model/model_multi_BiGRU_TCN_best.pkl")

        # k = 1
        # if (test_losses + k * test_losses_soil + abs(test_losses - k * test_losses_soil)) < best_validation_loss:
        #     best_test = test_losses
        #     best_test_soil = test_losses_soil
        #     best_validation_loss = ((test_losses + k * test_losses_soil) + 0.1 * abs(test_losses - k * test_losses_soil))
        #     early_stop_counter = 0
        #     torch.save(model, f"./forward_model/model_multi_test_best.pkl")
        # 早停判断
        # if test_losses < best_validation_loss:
        #     best_validation_loss = test_losses
        #     early_stop_counter = 0
        #     torch.save(model, f"./forward_model/model_multi_test5.pkl")
        # if test_losses_soil < best_validation_loss:
        #     best_validation_loss = test_losses_soil
        #     early_stop_counter = 0
        #     torch.save(model, f"./forward_model/model_multi_test_best.pkl")
        else:
            early_stop_counter += 1
        if early_stop_counter >= early_stop_patience:
            print(f"Early stopping at epoch {t + 1} as validation loss did not improve.")
            break
    # train_and_test_loss = np.c_[train_losses_, test_losses_]
    # header = "train_loss, test_loss"
    # np.savetxt(f"./train_results/train_and_test_loss_GRU_{force}.csv",
    #            train_and_test_loss, delimiter=",", header=header, comments='')
    # torch.save(model, f"./forward_model/model_multi_test_best.pkl")
    plot_train_test_loss(epochs, train_losses_, test_losses_)
    nni.report_final_result(best_test)
    nni.report_final_result(best_test_soil)
    nni.report_final_result(best_validation_loss)

    print(f"==============================\n model_multi_all train end and saved.")


if __name__ == "__main__":
    main()
