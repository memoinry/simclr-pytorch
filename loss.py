import argparse
import pandas as pd
import matplotlib.pyplot as plt

def generate_loss_curve(experiment_id):
    # 构建日志文件路径
    log_file_path = f'./logs/exman-train1.py/runs/{experiment_id}/logs.csv'

    try:
        # 读取 CSV 文件
        df = pd.read_csv(log_file_path)
    except FileNotFoundError:
        print(f"未找到文件: {log_file_path}")
        return

    # 假设损失列名为 'train_loss' 和 'test_loss'，你可能需要根据实际情况修改
    if 'train_loss' in df.columns:
        plt.figure(figsize=(10, 6))
        plt.plot(df['t'], df['train_loss'], label='Train Loss')
        if 'test_loss' in df.columns:
            plt.plot(df['t'], df['test_loss'], label='Test Loss')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title('Training and Test Loss Curve')
        plt.legend()
        plt.grid(True)

        # 保存图像
        output_image_path = f'./logs/exman-train1.py/runs/{experiment_id}/loss_curve.png'
        plt.savefig(output_image_path)
        print(f"损失曲线已保存到: {output_image_path}")

        # 显示图像
        plt.show()
    else:
        print("CSV 文件中未找到 'train_loss' 列，请检查列名。")

if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='生成损失曲线')
    parser.add_argument('--id', type=str, required=True, help='实验 ID')

    # 解析命令行参数
    args = parser.parse_args()

    # 调用生成损失曲线的函数
    generate_loss_curve(args.id)