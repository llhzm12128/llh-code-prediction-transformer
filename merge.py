import json

# 输出文件路径
output_file = 'D:\\projects\\llh-code-prediction-transformer\\data\\new_tree_150k.json'

# 打开输出文件，准备逐行写入
with open(output_file, 'w', encoding='utf-8') as merged_file:
    # 读取第一个 JSON 文件并逐行写入
    with open('D:\\projects\\llh-code-prediction-transformer\\data\\new_tree_50k.json', 'r', encoding='utf-8') as file1:
        for line in file1:
            # 直接写入当前行（每一行已经是一个完整的 JSON 数据）
            merged_file.write(line)

    # 读取第二个 JSON 文件并逐行写入
    with open('D:\\projects\\llh-code-prediction-transformer\\data\\new_tree_100k.json', 'r', encoding='utf-8') as file2:
        for line in file2:
            # 直接写入当前行（每一行已经是一个完整的 JSON 数据）
            merged_file.write(line)

print(f"两个 JSON 文件已成功合并到 '{output_file}' 中，且每一行都是一个独立的 AST 数据。")