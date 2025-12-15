#引入必要的库
import deepquantum as dq
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
import os
import time

# 固定随机种子以确保实验可重复性
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 设置默认数据类型为float32
torch.set_default_dtype(torch.float32)

#============================================
#1. 数据预处理
#============================================
def load_and_preprocess_data(train_dataset_path, test_dataset_path=None):
    """
    加载并预处理数据
    参数:
    train_dataset_path: 训练数据集路径
    test_dataset_path: 测试数据集路径（可选）
    返回:
    train_loader: 训练数据加载器
    scaler_X: X的归一化器
    scaler_Y: Y的归一化器
    X_test_normalized: 归一化后的测试数据特征（如果有测试集）
    Y_test_normalized: 归一化后的测试数据标签（如果有测试集）
    """
    # 1.1 加载训练数据集
    train_df = pd.read_csv(train_dataset_path)
    X = train_df.iloc[:, 0].values.reshape(-1, 1)  # 第一列x作为特征
    Y = train_df.iloc[:, 1].values.reshape(-1, 1)  # 第二列y作为标签，确保是二维数组

    # 1.2 进行归一化
    # 对X归一化到[0,1]
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    X_normalized = scaler_X.fit_transform(X)
    # 对Y归一化到[-0.5,0.5]
    scaler_Y = MinMaxScaler(feature_range=(-0.5, 0.5))
    Y_normalized = scaler_Y.fit_transform(Y) 

    # 1.3 全部数据用于训练
    X_train = X_normalized
    Y_train = Y_normalized

    # 1.4 转换为Tensor、创建数据加载器
    # 转换为Tensor
    train_data = torch.tensor(X_train, dtype=torch.float32)
    train_targets = torch.tensor(Y_train, dtype=torch.float32)
    # 创建TensorDataset
    train_dataset = TensorDataset(train_data, train_targets)
    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    # 如果提供了测试数据集路径，也加载测试数据
    X_test_normalized = None
    Y_test_normalized = None
    if test_dataset_path:
        test_df = pd.read_csv(test_dataset_path)
        X_test = test_df.iloc[:, 0].values.reshape(-1, 1)  # 第一列x作为特征
        Y_test = test_df.iloc[:, 1].values.reshape(-1, 1)  # 第二列y作为标签
        
        # 使用训练集的scaler进行归一化
        X_test_normalized = scaler_X.transform(X_test)
        Y_test_normalized = scaler_Y.transform(Y_test)
    
    return train_loader, scaler_X, scaler_Y, X_test_normalized, Y_test_normalized

#============================================
#2. 光量子线路定义
#============================================
# 创建光量子线路
def circuit(inputs, params):
    # 处理输入数据形状
    if isinstance(inputs, torch.Tensor):
        input_value = inputs.item() if inputs.dim() == 0 else inputs.squeeze().item()
    else:
        input_value = inputs if np.isscalar(inputs) else inputs.item()
    
    # 创建光量子线路
    cir = dq.QubitCircuit(nqubit=6) 
    
    # 第一层：输入编码（模式0,1,2）
    # 利用inputs手动初始化固定参数
    cir.ry(0, inputs=[input_value*3])
    cir.ry(1, inputs=[input_value*9])
    cir.ry(2, inputs=[input_value])
    
    # 输入编码微调，使用params[0]-params[2]
    cir.ry(0, inputs=[params[0]])
    cir.ry(1, inputs=[params[1]])
    cir.ry(2, inputs=[params[2]])
    
    # 权重累加一阶
    # 使用params[3]-params[14] (共12个参数)
    # 实现CRY门操作
    cir.ry(3, controls=[2], inputs=[params[3]])
    cir.ry(3, controls=[1], inputs=[params[4]])
    cir.ry(3, controls=[0], inputs=[params[5]])
    cir.ry(3, inputs=[params[6]])
    
    cir.ry(4, controls=[2], inputs=[params[7]])
    cir.ry(4, controls=[1], inputs=[params[8]])
    cir.ry(4, controls=[0], inputs=[params[9]])
    cir.ry(4, inputs=[params[10]])
    
    cir.ry(5, controls=[2], inputs=[params[11]])
    cir.ry(5, controls=[1], inputs=[params[12]])
    cir.ry(5, controls=[0], inputs=[params[13]])
    cir.ry(5, inputs=[params[14]])
    
    # 固定角度旋转
    cir.ry(3, inputs=[0.5])
    cir.ry(4, inputs=[0.5])
    cir.ry(5, inputs=[0.5])
    
    # 最大调制，使用params[15]-params[20] (共6个参数)
    cir.ry(4, controls=[3], inputs=[params[15]])
    cir.ry(5, controls=[3], inputs=[params[16]])
    cir.ry(3, controls=[4], inputs=[params[17]])
    cir.ry(5, controls=[4], inputs=[params[18]])
    cir.ry(3, controls=[5], inputs=[params[19]])
    cir.ry(4, controls=[5], inputs=[params[20]])
    
    # 输出微调，使用params[21]-params[23] (共3个参数)
    cir.ry(3, inputs=[params[21]])
    cir.ry(4, inputs=[params[22]])
    cir.ry(5, inputs=[params[23]])
    
    # layer2 - 控制比特为3,4,5，目标比特为0,1,2
    # 使用DeepQuantum的多控制位量子门
    cir.y(0, controls=[3, 4, 5])
    cir.y(1, controls=[3, 4, 5])
    cir.y(2, controls=[3, 4, 5])
    
    # layer2输入编码
    cir.ry(0, inputs=[input_value*7])
    cir.ry(1, inputs=[input_value*17])
    cir.ry(2, inputs=[input_value])
    
    # layer2输入编码微调，使用params[24]-params[26] (共3个参数)
    cir.ry(0, inputs=[params[24]])
    cir.ry(1, inputs=[params[25]])
    cir.ry(2, inputs=[params[26]])
    
    # 固定角度旋转
    cir.ry(0, inputs=[-0.5])
    cir.ry(1, inputs=[-0.5])
    cir.ry(2, inputs=[-0.5])
    
    # layer2最大调制，使用params[27]-params[32] (共6个参数)
    cir.ry(1, controls=[0], inputs=[params[27]])
    cir.ry(2, controls=[0], inputs=[params[28]])
    cir.ry(0, controls=[1], inputs=[params[29]])
    cir.ry(2, controls=[1], inputs=[params[30]])
    cir.ry(0, controls=[2], inputs=[params[31]])
    cir.ry(1, controls=[2], inputs=[params[32]])
    
    # layer2输出微调，使用params[33]-params[35] (共3个参数)
    cir.ry(0, inputs=[params[33]])
    cir.ry(1, inputs=[params[34]])
    cir.ry(2, inputs=[params[35]])
    
    # 测量
    cir.observable(0)
    cir.observable(1)
    cir.observable(2)

    cir()
    # 获取000态和111态的概率
    prob_000 = cir.get_prob(bits='000', wires=[0,1,2])
    prob_111 = cir.get_prob(bits='111', wires=[0,1,2])
    
    return prob_000, prob_111

#============================================
# 3. 光量子神经网络模型类定义
#============================================
class PhotonicNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # 初始化变分参数，每个参数在[0, 2*pi]之间
        self.params = nn.Parameter(torch.rand(36, dtype=torch.float32) * 2 * torch.pi)

    def forward(self, inputs):
        # 在前向过程中，变分参数作为data加入量子线路
        prob_000, prob_111 = circuit(inputs, self.params)
        # 返回两个概率的差值作为输出
        output = (prob_111 - prob_000) * 2
        return output

#============================================
# 4. 训练过程
#============================================
def train_model(train_loader, num_epochs=25, learning_rate=0.1, step_size=5, gamma=0.5):
    """
    训练模型
    参数:
    train_loader: 训练数据加载器
    num_epochs: 训练轮数
    learning_rate: 学习率
    step_size: 学习率衰减步长
    gamma: 学习率衰减因子
    返回:
    model: 训练好的模型
    train_losses: 训练损失列表
    training_time: 训练耗时（秒）
    """
    # 记录开始时间
    start_time = time.time()
    
    # 创建模型实例
    model = PhotonicNeuralNetwork()
    
    # 定义损失函数和优化器
    criterion = nn.MSELoss()  
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 
    # 学习率衰减
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    # 记录训练损失
    train_losses = []
    
    print("开始训练...")
    for epoch in range(num_epochs):
        # 训练模式
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for data, target in train_loader:
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播
            outputs = []
            for i in range(data.size(0)):
                input_val = data[i].squeeze() if data[i].dim() > 0 else data[i]
                output = model(input_val)
                outputs.append(output)
            
            outputs = torch.stack(outputs).unsqueeze(1)
            
            # 计算损失
            loss = criterion(outputs, target)
            # 反向传播
            loss.backward()
            
            # 更新参数
            optimizer.step()
            train_loss += loss.item()
            train_batches += 1
        scheduler.step()
        # 计算平均训练损失
        avg_train_loss = train_loss / train_batches
        train_losses.append(avg_train_loss)
        
        # 打印训练进度
        if (epoch + 1) % 1 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.6f}')
    
    # 记录结束时间并计算训练耗时
    end_time = time.time()
    training_time = end_time - start_time
    
    print("训练完成!")
    print(f"训练耗时: {training_time:.2f} 秒")
    return model, train_losses, training_time

#============================================
# 5. 模型评估
#============================================
def evaluate_model(model, X_test, Y_test, scaler_Y=None, use_real_values=True):
    """
    逐个样本进行测试评估
    参数:
    model: 训练好的模型
    X_test: 测试数据特征
    Y_test: 测试数据标签
    scaler_Y: Y的归一化器，用于反归一化（可选）
    use_real_values: 是否使用真实值（反归一化后的值）进行评估和绘图
    返回:
    predictions: 预测结果
    actuals: 真实标签
    mse: 均方误差
    rmse: 均方根误差
    predictions_original: 反归一化后的预测结果（如果use_real_values为True）
    actuals_original: 反归一化后的真实标签（如果use_real_values为True）
    """
    model.eval()
    predictions = []
    actuals = []
    
    # 逐个样本进行测试
    with torch.no_grad():
        for i in range(len(X_test)):
            # 获取单个样本
            data_point = torch.tensor(X_test[i], dtype=torch.float32)
            target_point = torch.tensor(Y_test[i], dtype=torch.float32)
            
            # 确保输入数据是标量
            input_val = data_point.squeeze() if data_point.dim() > 0 else data_point
            output = model(input_val)
            
            predictions.append(output.cpu().numpy())
            actuals.append(target_point.cpu().numpy())
    
    # 计算评估指标
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # 如果需要使用真实值（反归一化后的值）
    if use_real_values and scaler_Y is not None:
        # 反归一化预测结果和真实标签
        predictions_original = scaler_Y.inverse_transform(predictions.reshape(-1, 1)).flatten()
        actuals_original = scaler_Y.inverse_transform(actuals.reshape(-1, 1)).flatten()
        
        # 使用反归一化后的值计算评估指标
        mse = np.mean((predictions_original - actuals_original) ** 2)
        rmse = np.sqrt(mse)
        
        print(f"测试集评估结果 (使用真实值):")
        print(f"MSE: {mse:.6f}")
        print(f"RMSE: {rmse:.6f}")
        
        return predictions, actuals, mse, rmse, predictions_original, actuals_original
    else:
        # 使用归一化后的值计算评估指标
        mse = np.mean((predictions - actuals) ** 2)
        rmse = np.sqrt(mse)
        
        print(f"测试集评估结果:")
        print(f"MSE: {mse:.6f}")
        print(f"RMSE: {rmse:.6f}")
        
        return predictions, actuals, mse, rmse, None, None

#============================================
# 6. 结果可视化和保存
#============================================
def plot_training_loss(train_losses, save_path='training_loss_curve.png'):
    """
    绘制并保存训练损失曲线
    参数:
    train_losses: 训练损失列表
    save_path: 保存路径
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # 关闭图形以释放内存
    print(f"训练损失曲线已保存到: {save_path}")

def plot_predictions_vs_actuals(predictions, actuals, save_path='predictions_vs_actuals.png', title="Predictions vs Actual Values"):
    """
    绘制预测结果与真实值的对比图
    参数:
    predictions: 预测结果
    actuals: 真实标签
    save_path: 保存路径
    title: 图表标题
    """
    plt.figure(figsize=(10, 6))
    # 绘制真实数据散点图
    plt.scatter(range(len(actuals)), actuals, label='Actual Values', alpha=0.7, color='blue')
    # 绘制预测结果曲线
    plt.plot(predictions, label='Predicted Values', color='red', linewidth=1)
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"预测结果对比图已保存到: {save_path}")

def plot_predictions_vs_actuals_real_values(predictions_original, actuals_original, save_path='predictions_vs_actuals_real_values.png'):
    """
    绘制反归一化后预测结果与真实值的对比图
    参数:
    predictions_original: 反归一化后的预测结果
    actuals_original: 反归一化后的真实标签
    save_path: 保存路径
    """
    plt.figure(figsize=(10, 6))
    # 绘制真实数据散点图
    plt.scatter(range(len(actuals_original)), actuals_original, label='Actual Values (Real)', alpha=0.7, color='blue')
    # 绘制预测结果曲线
    plt.plot(predictions_original, label='Predicted Values (Real)', color='red', linewidth=1)
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.title('Predictions vs Actual Values (Real Values)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"反归一化预测结果对比图已保存到: {save_path}")

def save_predictions_to_csv(predictions, actuals, save_path='predictions_vs_actuals.csv'):
    """
    保存预测结果和真实标签到CSV文件
    参数:
    predictions: 预测结果
    actuals: 真实标签
    save_path: 保存路径
    """
    results_df = pd.DataFrame({
        'predictions': np.array(predictions).flatten(),
        'actuals': np.array(actuals).flatten()
    })
    results_df.to_csv(save_path, index=False)
    print(f"预测结果和真实标签已保存到: {save_path}")

def save_predictions_to_csv_real_values(predictions_original, actuals_original, save_path='predictions_vs_actuals_real_values.csv'):
    """
    保存反归一化后预测结果和真实标签到CSV文件
    参数:
    predictions_original: 反归一化后的预测结果
    actuals_original: 反归一化后的真实标签
    save_path: 保存路径
    """
    results_df = pd.DataFrame({
        'predictions_real': np.array(predictions_original).flatten(),
        'actuals_real': np.array(actuals_original).flatten()
    })
    results_df.to_csv(save_path, index=False)
    print(f"反归一化预测结果和真实标签已保存到: {save_path}")
#============================================
# 7. 主程序入口
#============================================
def main(train_dataset_path='Datasets/train_datasets/train_dataset_6.csv', 
         test_dataset_path='Datasets/test_datasets/train_dataset_6.csv',
         base_save_dir='Result',
         use_real_values=True):
    """
    主程序入口
    参数:
    train_dataset_path: 训练数据集路径
    test_dataset_path: 测试数据集路径
    base_save_dir: 基础结果保存目录
    use_real_values: 是否使用真实值（反归一化后的值）进行评估和绘图
    """
    # 从训练数据集路径中提取数据集编号
    import re
    dataset_match = re.search(r'train_dataset_(\d+)\.csv', train_dataset_path)
    if dataset_match:
        dataset_number = dataset_match.group(1)
        save_dir = os.path.join(base_save_dir, f'train_dataset_{dataset_number}')
        weights_filename = f'weights_{dataset_number}.pt'
    else:
        # 如果无法提取编号，使用默认名称
        save_dir = os.path.join(base_save_dir, 'train_dataset_unknown')
        weights_filename = 'weights_unknown.pt'
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 加载并预处理数据
    train_loader, scaler_X, scaler_Y, X_test_normalized, Y_test_normalized = load_and_preprocess_data(
        train_dataset_path, test_dataset_path)
    
    # 训练模型
    trained_model, train_losses, training_time = train_model(train_loader)
    
    # 评估模型
    if X_test_normalized is not None and Y_test_normalized is not None:
        # 使用evaluate_model函数进行评估，可以选择是否使用真实值
        result = evaluate_model(trained_model, X_test_normalized, Y_test_normalized, scaler_Y, use_real_values)
        predictions, actuals, mse, rmse, predictions_original, actuals_original = result
        
        # 保存结果
        loss_curve_path = os.path.join(save_dir, 'training_loss_curve.png')
        plot_training_loss(train_losses, loss_curve_path)
        
        # 如果使用真实值，则绘制和保存反归一化后的图表
        if use_real_values and predictions_original is not None and actuals_original is not None:
            # 保存反归一化后的预测结果和真实标签
            predictions_plot_path = os.path.join(save_dir, 'predictions_vs_actuals_real_values.png')
            plot_predictions_vs_actuals_real_values(predictions_original, actuals_original, predictions_plot_path)
            
            # 保存反归一化后的预测结果到CSV
            predictions_csv_path = os.path.join(save_dir, 'predictions_vs_actuals_real_values.csv')
            save_predictions_to_csv_real_values(predictions_original, actuals_original, predictions_csv_path)
            
            # 也保存归一化后的结果
            predictions_plot_path_norm = os.path.join(save_dir, 'predictions_vs_actuals_normalized.png')
            plot_predictions_vs_actuals(predictions, actuals, predictions_plot_path_norm, "Predictions vs Actual Values (Normalized)")
            
            predictions_csv_path_norm = os.path.join(save_dir, 'predictions_vs_actuals_normalized.csv')
            save_predictions_to_csv(predictions, actuals, predictions_csv_path_norm)
        else:
            # 只保存归一化后的结果
            predictions_plot_path = os.path.join(save_dir, 'predictions_vs_actuals.png')
            plot_predictions_vs_actuals(predictions, actuals, predictions_plot_path)
            
            predictions_csv_path = os.path.join(save_dir, 'predictions_vs_actuals.csv')
            save_predictions_to_csv(predictions, actuals, predictions_csv_path)
        
        # 保存模型权重文件（使用指定格式）
        model_weights_path = os.path.join(save_dir, weights_filename)
        torch.save(trained_model.state_dict(), model_weights_path)
        print(f"模型权重已保存到: {model_weights_path}")
        
        # 修改.pth为.pt
        model_path = os.path.join(save_dir, 'photonic_neural_network.pt')
        torch.save(trained_model.state_dict(), model_path)
        print(f"模型已保存到: {model_path}")
        
        # 根据是否使用真实值返回不同的结果
        if use_real_values and predictions_original is not None and actuals_original is not None:
            return trained_model, train_losses, training_time, predictions, actuals, mse, rmse, predictions_original, actuals_original
        else:
            return trained_model, train_losses, training_time, predictions, actuals, mse, rmse
    else:
        # 只保存训练损失曲线
        loss_curve_path = os.path.join(save_dir, 'training_loss_curve.png')
        plot_training_loss(train_losses, loss_curve_path)
        
        # 保存模型权重文件（使用指定格式）
        model_weights_path = os.path.join(save_dir, weights_filename)
        torch.save(trained_model.state_dict(), model_weights_path)
        print(f"模型权重已保存到: {model_weights_path}")
        
        # 修改.pth为.pt
        model_path = os.path.join(save_dir, 'photonic_neural_network.pt')
        torch.save(trained_model.state_dict(), model_path)
        print(f"模型已保存到: {model_path}")
        
        return trained_model, train_losses, training_time

if __name__ == "__main__":
    # 运行主程序
    main()