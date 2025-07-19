import json

input_jsonl = "LJSpeech_len_free/asru_nemo/LJSpeech_train_new_distortion_20_percen_SNR_10db_loss_weight_1-1_div_23.570146560668945_x.json"
output_txt = input_jsonl.replace('asru_nemo', 'asru').replace('.json', '.txt')

with open(input_jsonl, "r", encoding="utf-8") as fin, open(output_txt, "w", encoding="utf-8") as fout:
    for line in fin:
        data = json.loads(line)
        audio_filepath = data["audio_filepath"]
        text = data["text"]
        weight = data["loss_weight"]
        segment = data.get("loss_weight_segment")

        # 基本格式
        output_line = f"{audio_filepath}|{text}|{weight}"

        # 如果有 segment，加入
        if segment is not None:
            # 將 list 轉為字串，例如 [1.0, 0.5] -> "1.0,0.5"
            segment_str = ",".join(str(s) for s in segment)
            output_line += f"|{segment_str}"

        fout.write(output_line + "\n")
