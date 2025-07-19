import json
import os
from enum import Enum
import numpy as np
from collections import defaultdict
import pandas as pd
from tqdm import tqdm

class Action(Enum):
    JSON2TXT = "json2txt"
    ADD_UNIFORM_WEIGHT = "add_uniform_weight"
    ADD_SVDD_WEIGHT = "add_svdd_weight"
    FILTER_ZERO_ROWS = "filter_zero_rows"
    REPLACE_TXT = "replace_txt"
    ADD_FRAME_WEIGHT= "add_frame_weight"

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
            # svdd_dict[filename.replace('_DeepFilterNet3', '')] = float(distance)
            svdd_dict[filename] = float(distance)
    
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    if alpha == -1:
        alpha = svdd_dict[max(svdd_dict, key=svdd_dict.get)]
    output_file = output_file.replace('-1', f'-1({alpha})')
    print(f'alpha: {alpha} :{type(alpha)}')
    # print(f'svdd_dict; {svdd_dict}')
    
    # 在每一行末尾加上 '|1.0' 並寫回到新的文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in lines:
            filename = get_filename(line.split('|')[0])
            if filename not in svdd_dict:
                # print(f'key {filename} not found')
                print(f'file {filename}_DeepFilterNet3.wav not found')
                # a = 0
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
            modified_line = line.rstrip('\n') + '|[0, 1, 1]\n'
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
    
def filter_zero_rows(input_file, output_dir, output_txt):
    """
    讀取輸入文件，移除最後一列是 '0.0' 的行，
    並將結果寫入新的文件
    
    Args:
        input_file (str): 輸入文件路徑
        output_file (str): 輸出文件路徑
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 檢查或創建 TXT 文件的路徑
    output_txt_path = os.path.join(output_dir, output_txt)
    
    # 讀取並處理文件
    with open(input_file, 'r', encoding='utf-8') as f:
        # 讀取所有行
        lines = f.readlines()
        
        # 過濾掉最後一個值是 0.0 的行
        filtered_lines = []
        for line in lines:
            # 去除行尾的換行符並分割
            parts = line.strip().split('|')
            
            # 檢查是否有至少一個部分
            if parts and parts[-1] != '0.0':
                filtered_lines.append(line)
    
    # 寫入新文件
    with open(output_txt_path, 'w', encoding='utf-8') as f:
        f.writelines(filtered_lines)
  
def replace_txt(input_file, output_dir, output_txt, threshold, odd_text, new_text):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 檢查或創建 TXT 文件的路徑
    output_txt_path = os.path.join(output_dir, output_txt)
    
    with open(input_file, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()
    
    processed_lines = []
    for line in lines:
        parts = line.strip().split('|')
        if len(parts) == 3:
            path, text, score = parts
            score = float(score)
            if score >= threshold:
                path = path.replace(odd_text,new_text)  # 替換 path
            processed_lines.append(f"{path}|{text}|{score}")
    
    # 寫回結果到輸出文件
    with open(output_txt_path, 'w', encoding='utf-8') as outfile:
        outfile.write('\n'.join(processed_lines))

# chould get len from mos csv
# /mnt/md1/user_wago/MOS/csv/matbn_enhanced.csv
def add_frame_weight(input_file, mos_csv, output_dir, output_txt, svdd_file, frame_alpha):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 檢查或創建 TXT 文件的路徑
    output_txt_path = os.path.join(output_dir, output_txt)

    with open(svdd_file, "r") as f:
        data = json.load(f)
        test_scores = data["test_scores"]
    
    # 建立字典來存儲結果
    svdd_dict = defaultdict(dict)

    for score, path in tqdm(test_scores, desc='Build svdd dict'):
        filename = os.path.basename(path)  # 取得檔名
        base_name, time_range = filename.rsplit("_", 1)  # 以最後一個 "_" 拆分
        time_range = time_range.replace(".wav", "")  # 去掉副檔名
        
        svdd_dict[base_name][time_range] = score  # 存入字典
    
    mos_csv = pd.read_csv(mos_csv)
    filename_to_len = {row["filename"]: row["len_in_sec"] for _, row in mos_csv.iterrows()}
    
    for base_name in tqdm(svdd_dict, desc="Add complete_len"):
        matching_keys = [key for key in filename_to_len if base_name in key]
        if matching_keys:
            svdd_dict[base_name]['complete_len'] = filename_to_len[matching_keys[0]]
    # for base_name in tqdm(svdd_dict, desc="Add complete_len"):
    #     row = mos_csv[mos_csv['filename'].str.contains(base_name, na=False)]
    #     if not row.empty:
    #         svdd_dict[base_name]['complete_len'] = row.iloc[0]['len_in_sec']
    
    # 印出 result 字典的前 3 項
    # for key, value in list(svdd_dict.items())[:3]:
    #     print(f"{key}: {value}")

    with open(input_file, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()
    
    # 處理 lines
    processed_lines = []
    for line in lines:
        parts = line.strip().split('|')
        if len(parts) == 3:
            path, text, score = parts
            filename = os.path.basename(path).rsplit('.', 1)[0]
            
            if filename in svdd_dict and 'complete_len' in svdd_dict[filename]:
                frame_weight = [svdd_dict[filename]['complete_len']]
                
                # 取得時間區間與對應值，轉換為數字並排序
                sorted_intervals = sorted(
                    [(int(start), int(end), 1 - np.clip(1/frame_alpha * svdd_dict[filename][key], 0, 1)) for key in svdd_dict[filename] if key != 'complete_len' for start, end in [key.split('-')]],
                    key=lambda x: x[0]
                )
                
                # 平坦化數據加入 frame_weight
                for start, end, value in sorted_intervals:
                    frame_weight.extend([start, end, value])
                
                processed_lines.append(f"{path}|{text}|{score}|{frame_weight}")
    
    with open(output_txt_path, 'w', encoding='utf-8') as outfile:
        outfile.write('\n'.join(processed_lines))
            

def execute_action(action: Action, input_file, output_dir, MAX_LENGTH, svdd_file = None, replace_th=0.4, mos_csv=None):
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
            alpha = 0.8
            output_txt = os.path.basename(input_file).replace('.txt', f'_weight_alpha_{alpha}_aishell3_vq.txt')      # 最終輸出的 TXT 文件
            add_svdd_weight(input_file, svdd_file, output_dir, output_txt, alpha)
    elif action == Action.FILTER_ZERO_ROWS:
        output_txt = os.path.basename(input_file).replace('.txt', f'_remove_zero_row.txt')      # 最終輸出的 TXT 文件
        filter_zero_rows(input_file, output_dir, output_txt)
    elif action == Action.REPLACE_TXT:
        output_txt = os.path.basename(input_file).replace('.txt', f'_replace_txt_th_{replace_th}.txt')      # 最終輸出的 TXT 文件
        replace_txt(input_file, output_dir, output_txt, replace_th, "/enhanced/", "/default/")
        replace_txt(os.path.join(output_dir, output_txt), output_dir, output_txt, replace_th, "_DeepFilterNet3.wav", ".wav")
    elif action == Action.ADD_FRAME_WEIGHT:
        frame_alpha = 2.5
        output_txt = os.path.basename(input_file).replace('.txt', f'_frame_weight_{frame_alpha}_5_split.txt')      # 最終輸出的 TXT 文件
        add_frame_weight(input_file, mos_csv, output_dir, output_txt, svdd_file, frame_alpha)
    else :
        print("Unknown action")
    
if __name__ == '__main__':
    # input_json = '/mnt/md1/user_wago/data/matbn/matbn_test.json'  # 替換為你的 JSON 文件路徑
    # input_txt = 'data/matbn/matbn_train._less_17.txt'
    output_dir = 'data/matbn'    # 替換為你想要的輸出目錄
    MAX_LENGTH = 17
    svdd_file = '/mnt/md1/user_wago/SVDD/log_LJ_wt_aishell3_vq/matbn_enhanced_less_17_wt_vqscore_aishell3.json'
    # execute_action(Action.JSON2TXT, input_json, output_dir, MAX_LENGTH)
    
    # input_txt = 'data/matbn/matbn_train_enhanced._less_17.txt'
    # execute_action(Action.ADD_SVDD_WEIGHT, input_txt, output_dir, MAX_LENGTH, svdd_file)
    
    
    input_txt = 'data/matbn/default_weight_alpha_10._less_17.txt'
    execute_action(Action.FILTER_ZERO_ROWS, input_txt, output_dir, MAX_LENGTH, svdd_file)
    
    # input_txt = 'data/matbn/matbn_dev.txt'
    # execute_action(Action.ADD_UNIFORM_WEIGHT, input_txt, output_dir, MAX_LENGTH, svdd_file)
    
    # input_txt = 'data/matbn/matbn_denoisy_train._less_17_weight_alpha_1_all_mos_features.txt'
    # execute_action(Action.REPLACE_TXT, input_txt, output_dir, MAX_LENGTH, replace_th=0.4)
    
    # input_txt = 'data/matbn/matbn_denoisy_train._less_17_weight_alpha_1_all_mos_features_remove_zero_row.txt'
    # mos_csv = '/mnt/md1/user_wago/MOS/csv/matbn_enhanced.csv'
    # frame_svdd = '/mnt/md1/user_wago/SVDD/log_LJ_split_5/matbn_enhanced_split_5s.json'
    # execute_action(Action.ADD_FRAME_WEIGHT, input_txt, output_dir, MAX_LENGTH, svdd_file=frame_svdd, mos_csv=mos_csv)
    
    # input_txt = 'data/matbn/matbn_dev_weight_unifrom.txt'
    # execute_action(Action.ADD_UNIFORM_WEIGHT, input_txt, output_dir, MAX_LENGTH, svdd_file)

