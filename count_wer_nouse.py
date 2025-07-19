import csv
import glob
import os
import re
import string

import whisper
from jiwer import wer
from tqdm import tqdm

# === 設定路徑 ===
# eval_audios/2024-08-31_11-55-39_FastPitch_weight--val_loss=0
# eval_audios/2024-09-01_09-57-35_FastPitch_weight--val_loss=0_epoch_199
# eval_audios/2024-09-04_10-08-24_FastPitch_weight_segment--val_loss=0
# eval_audios/2024-09-08_12-12-01_FastPitch_weight_segment--val_loss=0

# eval_audios/2024-09-05_08-44-47_FastPitch_weight--val_loss=0
# eval_audios/2024-09-06_14-19-43_FastPitch_weight_segment--val_loss=0
wav_dir = "eval_audios/2024-09-06_14-19-43_FastPitch_weight_segment--val_loss=0"
tsv_path = "/mnt/md1/user_wago/data/LibriTTS/eval_sentences10.tsv"
output_root = "asr_result/wer_lower_only"

# === 初始化模型 ===
model = whisper.load_model("large")

# # === 標準化文字（移除標點） ===
# def normalize_text(text):
#     text = re.sub(rf"[{string.punctuation}]", "", text)
#     return text.lower()


def normalize_text(text):
    return text.strip().lower()


# === 讀取 TSV 標籤檔 ===
def get_text_dict(file_path):
    text_dict = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter='\t')
        for row in reader:
            if len(row) >= 2:
                key, value = row[0], row[1]
                text_dict[key] = normalize_text(value)
    return text_dict

labels = get_text_dict(tsv_path)

# === 建立輸出資料夾 ===
set_name = os.path.basename(os.path.normpath(wav_dir))
output_dir = os.path.join(output_root, set_name)
os.makedirs(output_dir, exist_ok=True)

# === 結果檔案路徑 ===
results_path = os.path.join(output_dir, "results.txt")
avg_wer_path = os.path.join(output_dir, "average_wer.txt")

# === 遞迴取得所有 .wav 檔案 ===
wav_files = glob.glob(os.path.join(wav_dir, "**/*.wav"), recursive=True)

# === 執行轉錄與計算 WER ===
total_wer = 0
count = 0

with open(results_path, 'w', encoding='utf-8') as out_f:
    for wav_path in tqdm(wav_files):
        fname = os.path.basename(wav_path)
        key = os.path.splitext(fname)[0]

        if key not in labels:
            print(f"[警告] 找不到對應標籤: {key}")
            continue

        # 推論轉錄
        result = model.transcribe(wav_path, language="en")
        pred_text = normalize_text(result["text"])
        ref_text = labels[key]

        sample_wer = wer(ref_text, pred_text)
        total_wer += sample_wer
        count += 1

        # 寫入結果
        out_f.write(f"{fname}\n")
        out_f.write(f"Reference: {ref_text}\n")
        out_f.write(f"Predicted: {pred_text}\n")
        out_f.write(f"WER: {sample_wer:.4f}\n\n")

        print(f"{fname} WER: {sample_wer:.4f}")

# === 平均 WER 結果 ===
with open(avg_wer_path, 'w', encoding='utf-8') as f:
    if count > 0:
        avg_wer = total_wer / count
        f.write(f"平均 WER: {avg_wer:.4f}\n")
        print(f"\n平均 WER: {avg_wer:.4f}")
    else:
        f.write("沒有有效樣本可評估。\n")
        print("沒有有效的樣本可以評估。")
