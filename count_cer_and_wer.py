import os
import json
import whisper
from jiwer import cer, wer
from tqdm import tqdm
from opencc import OpenCC
import string
import re
import csv

# 2024-11-04_15-50-17
# /mnt/md1/user_wago/Matcha-TTS/synth_output/all_test/2024-11-10_09-39-59
# /mnt/md1/user_wago/Matcha-TTS/synth_output/all_test/2024-11-13_15-17-19
# /mnt/md1/user_wago/Matcha-TTS/synth_output/all_test/2025-02-05_05-46-27

# "2025-05-24_23-13-04_Matcha-SW-BNG-400epoch.csv"
# "2025-05-24_23-07-15_Matcha-UW-BNG-400epoch.csv"

# van-CLN-400epoch: /mnt/md1/user_wago/Matcha-TTS/synth_output/LJ_output/2025-05-20_16-35-09
# van-BNG-200epoch: /mnt/md1/user_wago/Matcha-TTS/synth_output/LJ_output/2025-05-18_04-14-03

# UW-BNG-200epoch: /mnt/md1/user_wago/Matcha-TTS/synth_output/LJ_output/2025-05-15_10-27-12
# SW-BNG-200epoch: /mnt/md1/user_wago/Matcha-TTS/synth_output/LJ_output/2025-05-16_18-13-34
# UW-BNG-400epoch: /mnt/md1/user_wago/Matcha-TTS/synth_output/LJ_output/2025-05-24_23-07-15
# SW-BNG-400epoch: /mnt/md1/user_wago/Matcha-TTS/synth_output/LJ_output/2025-05-24_23-13-04

# /mnt/md1/user_wago/Matcha-TTS/synth_output/all_test/2025-07-15_19-18-23
wav_dir = "/mnt/md1/user_wago/Matcha-TTS/synth_output/all_test/2025-07-15_19-18-23"

# UW/CLN: /mnt/md1/user_wago/Matcha-TTS/synth_output/LJ_output/2025-05-27_01-59-15
# UW/FNG: /mnt/md1/user_wago/Matcha-TTS/synth_output/LJ_output/2025-05-30_14-35-00


# SW/CLN: /mnt/md1/user_wago/Matcha-TTS/synth_output/LJ_output/2025-06-01_00-05-35
# SW/BNG: /mnt/md1/user_wago/Matcha-TTS/synth_output/LJ_output/2025-05-16_18-13-34
# SW/FNG: /mnt/md1/user_wago/Matcha-TTS/synth_output/LJ_output/2025-05-29_11-21-06


# van/FNG: /mnt/md1/user_wago/Matcha-TTS/synth_output/LJ_output/2025-06-02_10-33-32

# van/CLN: /mnt/md1/user_wago/NeMo/eval_audios/2024-05-10_16-27-55_no_weight_clean_FastPitch_weight--val_loss=0
# van/BNG: /mnt/md1/user_wago/NeMo/eval_audios/2024-05-16_10-43-04_FastPitch_weight--val_loss=0
# van/FNG: /mnt/md1/user_wago/NeMo/eval_audios/2024-08-08_22-00-03_FastPitch_weight--val_loss=0

is_en = True

if is_en:
    tsv_path = "/mnt/md1/user_wago/data/LibriTTS/eval_sentences10.tsv"
    output_root = "asr_result/wer"
else:
    output_root = "asr_result/cer"
    jsonl_path = "data/matbn/json/matbn_test.json"

# === 初始化 ===
model = whisper.load_model("large")
if is_en == False:
    cc = OpenCC('s2t')  # 簡體轉繁體

# === 建立標準化文字的函數 ===
def normalize_text(text, is_en):
    if is_en:
        return text.strip().lower()
    else:
        text = re.sub(rf"[{string.punctuation}、，。！？；：「」『』（）《》〈〉…—～·\s]", "", text)
        return cc.convert(text)

def get_text_dict(file_path):
    text_dict = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            if len(row) >= 2:
                key, value = row[0], row[1]
                text_dict[key] = normalize_text(value,is_en)
    return text_dict

if is_en:
    labels = get_text_dict(tsv_path)
else:
    # === 讀取標籤 ===
    labels = {}
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            basename = os.path.basename(item["audio_path"])
            labels[basename] = normalize_text(item["text"],is_en)

# === 建立輸出資料夾 ===
set_name = os.path.basename(os.path.normpath(wav_dir))  # 取 wav_dir 最後一層資料夾名
output_dir = os.path.join(output_root, set_name)
os.makedirs(output_dir, exist_ok=True)

# === 結果檔案路徑 ===
results_path = os.path.join(output_dir, "results.txt")
if is_en:
    avg_result_path = os.path.join(output_dir, "average_wer.txt")
else:
    avg_result_path = os.path.join(output_dir, "average_cer.txt")

# === 轉錄與計算 CER ===
total_error = 0
count = 0

with open(results_path, 'w', encoding='utf-8') as out_f:
    for fname in tqdm(os.listdir(wav_dir)):
        if not fname.lower().endswith(".wav"):
            continue
        wav_path = os.path.join(wav_dir, fname)

        # 推論
        if is_en:
            result = model.transcribe(wav_path, language="en")
            fname = os.path.splitext(fname)[0]
        else:
            result = model.transcribe(wav_path, language="zh")
        pred_text = normalize_text(result["text"], is_en)

        # 取得對應標籤
        if fname not in labels:
            print(f"[警告] 找不到對應標籤: {fname}")
            continue
        label_text = labels[fname]

        # 計算 CER
        if is_en:
            sample_error = wer(label_text, pred_text)
        else:
            sample_error = cer(label_text, pred_text)
        total_error += sample_error
        count += 1

        # 寫入結果
        out_f.write(f"{fname}\n")
        out_f.write(f"Reference: {label_text}\n")
        out_f.write(f"Predicted: {pred_text}\n")
        if is_en:
            error_type = "WER"
        else:
            error_type = "CER"
        out_f.write(f"{error_type}: {sample_error:.4f}\n\n")

        print(f"{fname} {error_type}: {sample_error:.4f}")

# === 寫入平均 CER ===
with open(avg_result_path, 'w', encoding='utf-8') as f:
    if count > 0:
        avg_error = total_error / count
        f.write(f"平均 {error_type}: {avg_error:.4f}\n")
        print(f"\n平均 {error_type}: {avg_error:.4f}")
    else:
        f.write("沒有有效樣本可評估。\n")
        print("沒有有效的樣本可以評估。")
