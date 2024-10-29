import json
import os

MAX_LENGTH = 17
# 設置輸入 JSON 文件和輸出 TXT 文件的路徑
input_file = '/mnt/md1/user_wago/data/matbn/default_weight_alpha_5.json'  # 替換為你的 JSON 文件路徑
output_dir = '.'    # 替換為你想要的輸出目錄
output_txt = input_file.replace('.json', f'._less_{MAX_LENGTH}.txt')      # 最終輸出的 TXT 文件

# 檢查輸出目錄是否存在，不存在則創建
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 檢查或創建 TXT 文件的路徑
output_txt_path = os.path.join(output_dir, output_txt)

# 讀取每一行，並將所需內容寫入同一個 TXT 文件
with open(input_file, 'r') as f, open(output_txt_path, 'w', encoding='utf-8') as out_f:
    for i, line in enumerate(f):
        try:
            # 將每一行作為 JSON 加載
            json_data = json.loads(line)

            # 獲取每行 JSON 中的 `audio_path` 和 `text`
            audio_path = json_data['audio_path'].replace('wav/default', '/mnt/md1/user_wago/data/matbn/wav/default')
            text = json_data['ipa']
            if "loss_weight" in json_data:
                loss_weight = json_data['loss_weight']

            if text is not None and (json_data['duration'] < MAX_LENGTH or MAX_LENGTH == -1):
                if "loss_weight" in json_data:
                    out_f.write(f'{audio_path}|{text}|{loss_weight}\n')
                else:
                    # 將結果寫入到 txt 文件中，格式為 'audio_path|text'
                    out_f.write(f'{audio_path}|{text}\n')

        except json.JSONDecodeError:
            print(f"Error decoding JSON on line {i+1}")
        except KeyError:
            print(f"Missing required keys in JSON on line {i+1}")

print(f"All rows have been processed and saved to {output_txt_path}")

# input_file = '/mnt/md1/user_wago/Matcha-TTS/data/matbn/matbn_dev.txt'
# output_file = '/mnt/md1/user_wago/Matcha-TTS/data/matbn/matbn_dev_weight.txt'

# # 打開文件，讀取每一行並在行尾加上 '|1.0'
# with open(input_file, 'r', encoding='utf-8') as f:
#     lines = f.readlines()

# # 在每一行末尾加上 '|1.0' 並寫回到新的文件
# with open(output_file, 'w', encoding='utf-8') as f:
#     for line in lines:
#         # 去除末尾的換行符號並添加 '|1.0'
#         modified_line = line.rstrip('\n') + '|1.0\n'
#         f.write(modified_line)