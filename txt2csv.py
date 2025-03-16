import csv
def txt2csv():
    path_name = "FinancialPhraseBank/"
    names = ["Sentences_50Agree", "Sentences_66Agree", "Sentences_75Agree", "Sentences_AllAgree"]
    for name in names:
        # 定义输入和输出文件名
        input_file = path_name + name + '.txt'  # 替换为你的 TXT 文件名
        output_file = path_name + name + '.csv'  # 输出的 CSV 文件名

        # 打开输入文件并读取内容
        with open(input_file, 'r', encoding='ISO-8859-1') as infile:
            lines = infile.readlines()
        # 处理数据并写入 CSV 文件
        with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            # 写入 CSV 表头
            writer.writerow(['sentiment', 'text'])
            
            for line in lines:
                # 每行以 @ 分隔情感和文本
                if '@' in line:
                    text, sentiment = line.rsplit('@', 1)  # 从右侧分割
                    sentiment = sentiment.strip()  # 去掉前后空格
                    text = text.strip()  # 去掉前后空格
                    writer.writerow([sentiment, text])

        print(f"转换完成！CSV 文件已保存为 '{output_file}'。")

if __name__ == "__main__":
    txt2csv()