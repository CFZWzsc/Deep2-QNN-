import os
import sys
import argparse
import pandas as pd
import numpy as np
from model import main as train_model

def calculate_mae(predictions, actuals):
    """计算平均绝对误差(MAE)"""
    return np.mean(np.abs(predictions - actuals))

def train_single_dataset(train_dataset_path, test_dataset_path, results_base):
    """训练单个数据集"""
    # 确保结果目录存在
    os.makedirs(results_base, exist_ok=True)
    
    # 获取数据集编号
    dataset_num = None
    if "train_dataset_" in train_dataset_path:
        try:
            dataset_num = int(train_dataset_path.split("train_dataset_")[1].split(".csv")[0])
        except:
            pass
    
    print(f"\n{'='*50}")
    if dataset_num:
        print(f"开始训练数据集 {dataset_num}")
    else:
        print(f"开始训练指定数据集")
    print(f"{'='*50}")
    
    # 检查数据集文件是否存在
    if not os.path.exists(train_dataset_path):
        print(f"错误: 训练数据集 {train_dataset_path} 不存在")
        return None
        
    if not os.path.exists(test_dataset_path):
        print(f"错误: 测试数据集 {test_dataset_path} 不存在")
        return None
    
    # 记录单个数据集训练开始时间
    import time
    dataset_start_time = time.time()
    
    try:
        # 调用model.py的main函数训练模型
        print(f"训练数据集: {train_dataset_path}")
        print(f"测试数据集: {test_dataset_path}")
        
        # 调用model.py的main函数
        result = train_model(
            train_dataset_path=train_dataset_path,
            test_dataset_path=test_dataset_path,
            base_save_dir=results_base
        )
        
        # 记录单个数据集训练结束时间
        dataset_end_time = time.time()
        dataset_training_time = dataset_end_time - dataset_start_time
        
        # 提取训练和测试结果
        if len(result) >= 7:  # 有7个返回值
            trained_model, train_losses, training_time, predictions, actuals, mse, rmse = result[:7]
            
            # 计算MAE
            mae = calculate_mae(np.array(predictions).flatten(), np.array(actuals).flatten())
            
            # 计算最终训练损失
            final_train_loss = train_losses[-1] if train_losses else None
            
            # 打印训练时间信息
            print(f"数据集训练时间: {training_time:.2f} 秒 (来自model.py)")
            print(f"脚本执行时间: {dataset_training_time:.2f} 秒 (来自train.py)")
            
            # 返回结果
            result_dict = {
                'dataset': dataset_num if dataset_num else 'custom',
                'test_mae': mae,
                'test_mse': mse,
                'test_rmse': rmse,
                'final_train_loss': final_train_loss,
                'training_time': training_time,  # 从model.py获取的训练时间
                'script_training_time': dataset_training_time  # 从train.py计算的训练时间
            }
            
        elif len(result) >= 6:  # 为了兼容旧版本
            trained_model, train_losses, predictions, actuals, mse, rmse = result[:6]
            
            # 计算MAE
            mae = calculate_mae(np.array(predictions).flatten(), np.array(actuals).flatten())
            
            # 计算最终训练损失
            final_train_loss = train_losses[-1] if train_losses else None
            
            # 返回结果
            result_dict = {
                'dataset': dataset_num if dataset_num else 'custom',
                'test_mae': mae,
                'test_mse': mse,
                'test_rmse': rmse,
                'final_train_loss': final_train_loss,
                'training_time': dataset_training_time,  # 使用从train.py计算的时间
                'script_training_time': dataset_training_time
            }
            
            # 打印训练时间信息
            print(f"数据集训练时间: {dataset_training_time:.2f} 秒")
        else:
            print(f"错误: 返回的结果数量不足")
            return None
        
        print(f"数据集训练完成!")
        return result_dict
        
    except Exception as e:
        print(f"训练数据集时发生错误: {str(e)}")
        return None

def train_all_datasets():
    """遍历训练所有十二个数据集"""
    # 数据集基础路径
    train_datasets_base = "Datasets/train_datasets"
    test_datasets_base = "Datasets/test_datasets"
    results_base = "Result"
    # 确保结果目录存在
    os.makedirs(results_base, exist_ok=True)
    
    # 创建用于存储所有结果的列表
    all_results = []
    
    # 记录总训练时间
    total_training_time = 0
    
    # 遍历12个数据集
    for i in range(1, 13):
        print(f"\n{'='*50}")
        print(f"开始训练数据集 {i}/12")
        print(f"{'='*50}")
        
        # 构建数据集路径
        train_dataset_path = os.path.join(train_datasets_base, f"train_dataset_{i}.csv")
        test_dataset_path = os.path.join(test_datasets_base, f"train_dataset_{i}.csv")
        
        # 检查数据集文件是否存在
        if not os.path.exists(train_dataset_path):
            print(f"警告: 训练数据集 {train_dataset_path} 不存在，跳过...")
            continue
            
        if not os.path.exists(test_dataset_path):
            print(f"警告: 测试数据集 {test_dataset_path} 不存在，跳过...")
            continue
        
        # 记录单个数据集训练开始时间
        import time
        dataset_start_time = time.time()
        
        try:
            # 调用model.py的main函数训练模型
            print(f"训练数据集: {train_dataset_path}")
            print(f"测试数据集: {test_dataset_path}")
            
            # 调用model.py的main函数
            result = train_model(
                train_dataset_path=train_dataset_path,
                test_dataset_path=test_dataset_path,
                base_save_dir=results_base
            )
            
            # 记录单个数据集训练结束时间
            dataset_end_time = time.time()
            dataset_training_time = dataset_end_time - dataset_start_time
            total_training_time += dataset_training_time
            
            # 提取训练和测试结果
            if len(result) >= 7:  # 有7个返回值
                trained_model, train_losses, training_time, predictions, actuals, mse, rmse = result[:7]
                
                # 计算MAE
                mae = calculate_mae(np.array(predictions).flatten(), np.array(actuals).flatten())
                
                # 记录最终训练损失
                final_train_loss = train_losses[-1] if train_losses else None
                
                # 将结果添加到总表
                all_results.append({
                    'dataset': i,
                    'test_mae': mae,
                    'test_mse': mse,
                    'test_rmse': rmse,
                    'final_train_loss': final_train_loss,
                    'training_time': training_time,  # 从model.py获取的训练时间
                    'script_training_time': dataset_training_time  # 从train.py计算的训练时间
                })
                
                # 打印训练时间信息
                print(f"数据集 {i} 训练时间: {training_time:.2f} 秒 (来自model.py)")
                print(f"数据集 {i} 脚本执行时间: {dataset_training_time:.2f} 秒 (来自train.py)")
            elif len(result) >= 6:  # 为了兼容旧版本
                trained_model, train_losses, predictions, actuals, mse, rmse = result[:6]
                
                # 计算MAE
                mae = calculate_mae(np.array(predictions).flatten(), np.array(actuals).flatten())
                
                # 记录最终训练损失
                final_train_loss = train_losses[-1] if train_losses else None
                
                # 将结果添加到总表
                all_results.append({
                    'dataset': i,
                    'test_mae': mae,
                    'test_mse': mse,
                    'test_rmse': rmse,
                    'final_train_loss': final_train_loss,
                    'training_time': dataset_training_time,  # 使用从train.py计算的时间
                    'script_training_time': dataset_training_time
                })
                
                # 打印训练时间信息
                print(f"数据集 {i} 训练时间: {dataset_training_time:.2f} 秒")
            else:
                print(f"警告: 数据集 {i} 返回的结果数量不足")
            
            print(f"数据集 {i} 训练完成!")
            
        except Exception as e:
            print(f"训练数据集 {i} 时发生错误: {str(e)}")
            print("继续下一个数据集...")
            continue
    
    # 创建总表CSV文件
    if all_results:
        summary_df = pd.DataFrame(all_results)
        summary_csv_path = os.path.join(results_base, "summary_results.csv")
        summary_df.to_csv(summary_csv_path, index=False)
        print(f"\n总表CSV文件已保存到: {summary_csv_path}")
        print("\n总表内容:")
        print(summary_df.to_string(index=False))
        
        # 打印总体训练时间
        print(f"\n总训练时间: {total_training_time:.2f} 秒")
    
    print(f"\n{'='*50}")
    print("所有数据集训练完成!")
    print(f"结果保存在: {results_base}")
    print(f"总训练时间: {total_training_time:.2f} 秒")
    print(f"{'='*50}")

def main():
    # 创建解析器
    parser = argparse.ArgumentParser(description='光量子神经网络训练脚本')
    
    # 添加参数
    parser.add_argument('--train-dataset', type=str, 
                        help='训练数据集路径')
    parser.add_argument('--test-dataset', type=str, 
                        help='测试数据集路径')
    parser.add_argument('--result-dir', type=str, default='Result',
                        help='结果保存目录 (默认: Result)')
    parser.add_argument('--all', action='store_true',
                        help='训练所有12个数据集')
    
    # 解析参数
    args = parser.parse_args()
    
    # 如果指定了--all参数，则训练所有数据集
    if args.all:
        train_all_datasets()
        return
    
    # 如果指定了训练集和测试集，则训练单个数据集
    if args.train_dataset and args.test_dataset:
        result = train_single_dataset(args.train_dataset, args.test_dataset, args.result_dir)
        if result:
            print("\n训练结果:")
            print(f"数据集: {result['dataset']}")
            print(f"测试集MAE: {result['test_mae']:.6f}")
            print(f"测试集MSE: {result['test_mse']:.6f}")
            print(f"测试集RMSE: {result['test_rmse']:.6f}")
            print(f"最终训练损失: {result['final_train_loss']:.6f}")
            print(f"训练时间: {result['training_time']:.2f} 秒")
            print(f"脚本执行时间: {result['script_training_time']:.2f} 秒")
        return
    
    # 如果没有指定参数，则训练所有数据集（保持向后兼容）
    if not args.train_dataset and not args.test_dataset:
        train_all_datasets()
        return
    
    # 如果只指定了其中一个参数，则报错
    parser.print_help()

if __name__ == "__main__":
    main()