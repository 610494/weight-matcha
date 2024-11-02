import json
import os
from enum import Enum
import numpy as np

class Action(Enum):
    JSON2TXT = "json2txt"
    ADD_UNIFORM_WEIGHT = "add_uniform_weight"
    ADD_SVDD_WEIGHT = "add_svdd_weight"

def get_filename(file):
    return os.path.basename(file).rsplit('.', 1)[0]

def add_svdd_weight(input_file, svdd_file, output_dir, output_file, alpha=-1):
    """
    input_file (str): input txt file w/o weight
    svdd_file (str): svdd result for wavs in input_file
    output_dir (str): output dir
    output_file (str): output file name
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, output_file)
    
    svdd_dict = {}
    with open(svdd_file, 'r') as f:
        svdd_distance_list = json.load(f)['test_scores']
        for svdd_distance in svdd_distance_list:
            distance, file = svdd_distance
            filename = get_filename(file)
            svdd_dict[filename.replace('_DeepFilterNet3', '')] = float(distance)
    
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    if alpha == -1:
        alpha = svdd_dict[max(svdd_dict, key=svdd_dict.get)]
    output_file = output_file.replace('-1', f'-1({alpha})')
    print(f'alpha: {alpha} :{type(alpha)}')
    
    # 在每一行末尾加上 '|1.0' 並寫回到新的文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in lines:
            filename = get_filename(line.split('|')[0])
            if filename not in svdd_dict:
                # print(f'key {filename} not found')
                print(f'/mnt/md1/user_wago/data/matbn/wav/enhanced/{filename}_DeepFilterNet3.wav')
            else:
                weight = 1 - np.clip(1/alpha * svdd_dict[filename], 0, 1)
            modified_line = line.rstrip('\n') + f'|{weight}\n'
            f.write(modified_line)

def add_uniform_weight(input_file, output_dir, output_file):
    """
    input_file (str): input txt file w/o weight
    output_dir (str): output dir
    output_file (str): output file name
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, output_file)
    
    # 打開文件，讀取每一行並在行尾加上 '|1.0'
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 在每一行末尾加上 '|1.0' 並寫回到新的文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in lines:
            # 去除末尾的換行符號並添加 '|1.0'
            modified_line = line.rstrip('\n') + '|1.0\n'
            f.write(modified_line)

def json2txt(input_file, output_dir, output_txt, MAX_LENGTH = 17):
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
    
def execute_action(action: Action, input_file, output_dir, MAX_LENGTH, svdd_file = None):
    """根據傳入的 action Enum 值執行相應的操作"""
    if action == Action.JSON2TXT:
        output_txt = os.path.basename(input_file).replace('.json', f'._less_{MAX_LENGTH}.txt')      # 最終輸出的 TXT 文件
        json2txt(input_file, output_dir, output_txt, MAX_LENGTH)
    elif action == Action.ADD_UNIFORM_WEIGHT:
        output_txt = os.path.basename(input_file).replace('.txt', f'_weight_unifrom.txt')      # 最終輸出的 TXT 文件
        add_uniform_weight(input_file, output_dir, output_txt)
    elif action == Action.ADD_SVDD_WEIGHT:
        if svdd_file == None:
            print("Unknown action")
        else:
            alpha = -1
            output_txt = os.path.basename(input_file).replace('.txt', f'_weight_alpha_{alpha}.txt')      # 最終輸出的 TXT 文件
            add_svdd_weight(input_file, svdd_file, output_dir, output_txt)
    else:
        print("Unknown action")
    
if __name__ == '__main__':
    input_json = 'data/matbn/json/matbn_train.json'  # 替換為你的 JSON 文件路徑
    input_txt = 'data/matbn/matbn_train._less_17.txt'
    output_dir = 'data/matbn'    # 替換為你想要的輸出目錄
    MAX_LENGTH = 17
    svdd_file = '/mnt/md1/user_wago/SVDD/log_LJ_len_free_nisqa/matbn_enhanced_len_free_nisqa.json'
    # execute_action(Action.JSON2TXT, input_json, output_dir, MAX_LENGTH)
    execute_action(Action.ADD_SVDD_WEIGHT, input_txt, output_dir, MAX_LENGTH, svdd_file)
    

