# 定義文件名
input_file = '/mnt/md1/user_wago/Matcha-TTS/data/matbn/default._less_17.txt'  # 原始文件名
output_file = '/mnt/md1/user_wago/Matcha-TTS/data/matbn/default._less_17_uni_weight.txt'  # 輸出文件名

# 讀取文件，並在每行後面加上 |1
with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
    for line in infile:
        # 去除行尾的換行符，並加上 |1
        new_line = line.rstrip('\n') + '|1\n'
        # 寫入新文件
        outfile.write(new_line)

print("處理完成，結果已寫入", output_file)
