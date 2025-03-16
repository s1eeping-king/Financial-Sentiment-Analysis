import pandas as pd

def count_len():
    # 定义输入文件名
    input_file = 'FinancialPhraseBank/all-data.csv'  # 替换为你的 CSV 文件名

    # 读取 CSV 文件
    df = pd.read_csv(input_file)

    # 检查是否存在 'text' 列
    if 'text' in df.columns:
        # 统计长度大于 128 的文本数量
        long_texts_count = df['text'].apply(lambda x: len(str(x)) > 315).sum()
        print(f"长度大于 315 的文本数量: {long_texts_count}")
        
        # 查找文本的最大长度
        max_length = df['text'].apply(lambda x: len(str(x))).max()
        print(f"文本的最大长度: {max_length}")
    else:
        print("CSV 文件中没有 'text' 列。")

if __name__ == "__main__":
    count_len()