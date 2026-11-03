import torch.nn as nn
try:
    import nni  # only needed for hyper-parameter search
except ImportError:
    nni = None
import torch.optim.lr_scheduler
from torch.utils.data import DataLoader
import torch.optim as optim
from Volume_prediction_model.Volume_model3_no_dis.data_load import *
import matplotlib.pyplot as plt
import torch.nn.functional as F

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"


class InceptionTimeChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=8):
        super(InceptionTimeChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc1 = nn.Conv1d(in_channels, in_channels // ratio, kernel_size=1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv1d(in_channels // ratio, in_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out) * x


class InceptionTimeSpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(InceptionTimeSpatialAttention, self).__init__()
        self.conv1 = nn.Conv1d(2, 1, kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        return self.sigmoid(out) * x


class InceptionTimeCBAM(nn.Module):
    def __init__(self, in_channels, ratio=8, kernel_size=7):
        super(InceptionTimeCBAM, self).__init__()
        self.channel_attention = InceptionTimeChannelAttention(in_channels, ratio)
        self.spatial_attention = InceptionTimeSpatialAttention(kernel_size)

    def forward(self, x):
        out = self.channel_attention(x)
        out = self.spatial_attention(out)
        return out


class InceptionModuleCBAM(nn.Module):
    def __init__(self, in_channels, out_channels, bottleneck_channels, kernel_sizes, stride=1, use_residual=True,
                 use_bottleneck=True):
        super(InceptionModuleCBAM, self).__init__()
        self.use_residual = use_residual
        self.use_bottleneck = use_bottleneck
        self.in_channels = in_channels

        if use_bottleneck:
            self.bottleneck = nn.Conv1d(in_channels, bottleneck_channels, kernel_size=(1,),
                                        stride=(stride,), bias=False).to(device)
            in_channels = bottleneck_channels

        self.conv_list = nn.ModuleList().to(device)
        for kernel_size in kernel_sizes:
            padding = (kernel_size - 1) // 2  # Same padding
            self.conv_list.append(
                nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=(1,), padding=padding,
                          bias=False)).to(device)

        # MaxPool followed by Conv to ensure size matching
        self.maxpool = nn.MaxPool1d(3, stride=stride, padding=1).to(device)
        self.conv_pool = nn.Conv1d(in_channels, out_channels, kernel_size=(1,),
                                   stride=(stride,), padding=0, bias=False).to(device)

        self.batch_norm = nn.BatchNorm1d(out_channels * (len(kernel_sizes) + 1)).to(device)
        self.out_channels = out_channels * (len(kernel_sizes) + 1)

        self.residual = nn.Conv1d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=(1,),
                                  stride=(1,), padding=0, bias=False).to(device)
        self.residual_bn = nn.BatchNorm1d(self.out_channels).to(device)

        self.cbam = InceptionTimeCBAM(out_channels * 4)

    def forward(self, x):
        original_x = x
        if self.use_bottleneck:
            x = self.bottleneck(x)

        conv_outputs = [conv(x) for conv in self.conv_list]
        maxpool_output = self.conv_pool(self.maxpool(x))
        conv_outputs.append(maxpool_output)

        x = torch.cat(conv_outputs, 1)
        x = self.cbam(x)

        x = self.batch_norm(x)

        if self.use_residual:
            if original_x.shape[1] != self.out_channels:
                original_x = self.residual(original_x)
                original_x = self.residual_bn(original_x)
            x += original_x

        x = F.relu(x)
        return x


class InceptionTimeRegressionCBAM(nn.Module):
    def __init__(self, output_dim, input_features, num_modules, out_channels, bottleneck_channels):
        super(InceptionTimeRegressionCBAM, self).__init__()
        self.inception_modules = nn.Sequential(
            *[InceptionModuleCBAM(input_features if i == 0 else out_channels * 4, out_channels, bottleneck_channels,
                                  [11, 21, 41], use_residual=True, use_bottleneck=True) for i in range(num_modules)]
        )
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(out_channels * 4, output_dim).to(device)

    def forward(self, x):
        # 转置输入以匹配 Conv1d 需求 [batch_size, features, seq_length]
        x = x.permute(0, 2, 1)
        x = self.inception_modules(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        # x = F.softmax(x, dim=1)
        return x


def train_model(train_loader_, model_, optimizer_, device_, data_size, criterion_, scheduler=None):
    model_.train()
    train_loss = 0.0
    for input_batch_, output_label_batch_ in train_loader_:
        input_batch_ = input_batch_.to(device_).to(torch.float32)  # 输入特征
        # input_batch_normalized_ = input_batch_normalized_.to(device_).to(torch.float32)  # 归一化后的输入特征
        output_label_batch_ = output_label_batch_.to(device_)  # 真实值
        optimizer_.zero_grad()
        outputs = model_(input_batch_).to(device_).to(torch.float32)  # 机器学习模型的输出
        loss = criterion_(outputs, output_label_batch_)
        loss.backward()
        optimizer_.step()
        # train_loss += loss.item()
        train_loss += loss.item() * output_label_batch_.shape[0] / data_size
    scheduler.step()
    # epoch_loss = train_loss / len(train_loader_)
    return train_loss


def test_model(test_loader_, model_, device_, data_size, criterion_):
    model_.eval()
    with torch.no_grad():
        test_loss = 0.0
        for input_batch_, output_label_batch_ in test_loader_:
            input_batch_ = input_batch_.to(device_).to(torch.float32)  # 输入特征
            output_label_batch_ = output_label_batch_.to(device_)  # 真实值
            outputs = model_(input_batch_).to(device_).to(torch.float32)  # 机器学习模型的输出
            loss = criterion_(outputs, output_label_batch_)
            # test_loss += loss.item()
            test_loss += loss.item() * output_label_batch_.shape[0] / data_size
        # epoch_loss = test_loss / len(test_loader_)
    return test_loss


def plot_train_test_loss(epochs_, train_losses_arr, test_losses_arr):
    plt.plot(range(epochs_), train_losses_arr, color=(40 / 255, 124 / 255, 183 / 255), linewidth=1, label='train')
    plt.plot(range(epochs_), test_losses_arr, color=(255 / 255, 156 / 255, 75 / 255), linewidth=1, label='test')
    legend = plt.legend(loc='upper right', ncol=2)
    legend.get_frame().set_alpha(0)  # 设置图例的透明度为0
    plt.show()


# 使用示例
def main():
    # 超参数
    params = {
        'batch_size': 32,
        'num_modules': 2,
        'out_channels': 32,
        'bottleneck_channels': 32,
        'learning_rate': 0.094893227068148,  # 学习率
    }

    optimized_params = nni.get_next_parameter()
    params.update(optimized_params)
    print(params)
    # 加载训练数据

    batch_size = params['batch_size']
    data_size_train = 360
    data_size_test = 90
    train_dataset = TrajectoryDataset(data_file="./../../OriginalData"
                                                "/All_material_data_excavation_3Classes_P_Check_excavation"
                                                "/Other_3_Classes"
                                                "/data_train_other_3_seq.npy")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # 加载测试数据
    test_dataset = TrajectoryDataset(data_file="./../../OriginalData"
                                               "/All_material_data_excavation_3Classes_P_Check_excavation"
                                               "/Other_3_Classes"
                                               "/data_test_other_3_seq.npy")
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 定义模型、损失函数和优化器
    model = InceptionTimeRegressionCBAM(output_dim=1, input_features=7, num_modules=params['num_modules'],
                                        out_channels=params['out_channels'],
                                        bottleneck_channels=params['bottleneck_channels'])
    criterion = nn.MSELoss()
    # criterion = nn.L1Loss()
    # optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=0.001)
    optimizer = torch.optim.AdamW(model.parameters(), lr=params['learning_rate'], betas=(0.9, 0.99))

    # 训练模型
    epochs = 300
    train_losses_ = np.zeros(epochs)
    test_losses = 0
    test_losses_ = np.zeros(epochs)

    best_validation_loss = float('inf')
    early_stop_patience = 500  # 五轮不下降触发
    early_stop_counter = 0
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=epochs // 3, gamma=0.1)
    for t in range(epochs):
        print(f"-------------------------------\nEpoch {t + 1}")
        train_losses = train_model(train_loader, model, optimizer, device, data_size_train, criterion, scheduler)
        test_losses = test_model(test_loader, model, device, data_size_test, criterion)
        train_losses_[t] = train_losses
        test_losses_[t] = test_losses
        nni.report_intermediate_result(train_losses)
        nni.report_intermediate_result(test_losses)

        if train_losses < 3.1 and test_losses < 3:
            break

        # 早停判断
        if test_losses < best_validation_loss:
            best_validation_loss = test_losses
            early_stop_counter = 0
        else:
            early_stop_counter += 1
        if early_stop_counter >= early_stop_patience:
            print(f"Early stopping at epoch {t + 1} as validation loss did not improve.")
            break
    train_and_test_loss = np.c_[train_losses_, test_losses_]
    header = "train_loss, test_loss"
    np.savetxt("./train_results/train_and_test_loss_Patch_InceptionTimeCBAM.csv",
               train_and_test_loss, delimiter=",", header=header, comments='')
    plot_train_test_loss(epochs, train_losses_, test_losses_)
    nni.report_final_result(test_losses)
    torch.save(model, "./model/model_Patch_InceptionTimeCBAM.pkl")


if __name__ == "__main__":
    main()
