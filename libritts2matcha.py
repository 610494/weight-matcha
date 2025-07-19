input_tsv_path = "/mnt/md1/user_wago/data/LibriTTS/eval_sentences10.tsv"
output_txt_path = "/mnt/md1/user_wago/Matcha-TTS/LJSpeech_len_free/asru/eval_sentences10.txt"

with open(input_tsv_path, "r", encoding="utf-8") as tsv_file, open(output_txt_path, "w", encoding="utf-8") as txt_file:
    for line in tsv_file:
        parts = line.strip().split("\t")
        if len(parts) != 3:
            continue  # 跳過格式不正確的行
        file_id, text, _ = parts
        formatted_line = f"{file_id}|{text}|1.0\n"
        txt_file.write(formatted_line)
