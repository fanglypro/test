import os
import json
import numpy as np
import torch
import soundfile as sf
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
import whisper
import re

AUDIO_FILE = "/root/autodl-tmp/0new/omni-time/audio/part/VM.mp3"
WORK_DIR = "/root/autodl-tmp/0new/omni-time/audio"
OUTPUT_JSON = os.path.join(WORK_DIR, "audio_events_with_subtitles.json")
SUMMARY_JSON = os.path.join(WORK_DIR, "audio_segments_summary.json")
SEGMENT_DIR = os.path.join(WORK_DIR, "whisper_segments")
os.makedirs(SEGMENT_DIR, exist_ok=True)

print("正在加载 Qwen2-Audio...")
qwen_model_path = "/root/autodl-tmp/model/Qwen2-Audio-7B-Instruct"
qwen_processor = AutoProcessor.from_pretrained(qwen_model_path)
qwen_model = Qwen2AudioForConditionalGeneration.from_pretrained(qwen_model_path, device_map="auto")
print("Qwen2-Audio 加载完成.")

print("正在加载 Whisper-large 到 CPU...")
whisper_model = whisper.load_model("large", device="cpu")
print("Whisper 加载完成.")

def clean_description(desc):
    # 去掉括号里全是数字、逗号、点、横线、空格的内容
    return re.sub(r"\((?:[\d\.,\- ]+)\)", "", desc).replace("  ", " ").strip()

def safe_write_wav(path, audio, sr):
    if len(audio) == 0:
        print(f"Warning: skip writing empty audio to {path}")
        return False
    sf.write(path, audio, sr)
    return True

def process_audio(audio_file, work_dir, out_json, summary_json):
    # 1. Whisper分段（直接用原始音频文件）
    result = whisper_model.transcribe(audio_file, language='en', fp16=False)
    segments = result["segments"]

    # 2. 读取原始音频
    y, sr = sf.read(audio_file)
    if len(y.shape) > 1:
        y = y[:, 0]  # 只取第一通道

    outputs = []
    last_end = 0
    for idx, seg in enumerate(segments):
        start_sec = seg["start"]
        end_sec = seg["end"]
        start = int(start_sec * sr)
        end = int(end_sec * sr)
        if end <= start:
            print(f"Warning: segment {idx} is empty (start={start}, end={end}), skipped.")
            continue
        audio_seg = y[start:end]
        seg_file = os.path.join(SEGMENT_DIR, f"segment_{idx}.wav")
        if not safe_write_wav(seg_file, audio_seg, sr):
            continue

        # 事件描述（Qwen2-Audio）
        import librosa
        target_sr = qwen_processor.feature_extractor.sampling_rate
        seg_data, _ = librosa.load(seg_file, sr=target_sr)
        conversation = [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {"role": "user", "content": [
                {"type": "audio", "audio_url": None},
                {"type": "text", "text": "Recognize all events in the audio and describe them in detail, but do not mention any time intervals or durations."},
            ]},
        ]
        text = qwen_processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        inputs = qwen_processor(text=text, audios=[seg_data], return_tensors="pt", padding=True)
        inputs["input_ids"] = inputs["input_ids"].to(qwen_model.device)
        inputs["input_features"] = inputs["input_features"].to(qwen_model.device)
        with torch.no_grad():
            generate_ids = qwen_model.generate(**inputs, max_length=256)
            generate_ids = generate_ids[:, inputs["input_ids"].size(1):]
            event_desc = qwen_processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        event_desc = clean_description(event_desc)

        # 字幕（Whisper精确转写分段wav）
        sub_result = whisper_model.transcribe(seg_file, language='en', fp16=False)
        subtitle = sub_result["text"].strip()

        outputs.append({
            "segment_index": idx,
            "start_sec": round(start_sec, 2),
            "end_sec": round(end_sec, 2),
            "description": event_desc,
            "subtitle": subtitle
        })
        print(f"{round(start_sec, 2)} s - {round(end_sec, 2)} s, audio events: {event_desc}. subtitle: {subtitle}")
        os.remove(seg_file)
        last_end = end_sec

    # 检查是否有未被分段覆盖的音频末尾部分（whisper常见“漏尾”问题），如有则补上
    total_duration = len(y) / sr
    if last_end < total_duration - 0.1:
        print(f"Warning: Whisper segments ended at {last_end:.2f}s, but audio ends at {total_duration:.2f}s. Auto-appending tail segment.")
        start = int(last_end * sr)
        end = len(y)
        audio_seg = y[start:end]
        seg_file = os.path.join(SEGMENT_DIR, f"segment_tail.wav")
        if safe_write_wav(seg_file, audio_seg, sr):
            import librosa
            target_sr = qwen_processor.feature_extractor.sampling_rate
            seg_data, _ = librosa.load(seg_file, sr=target_sr)
            conversation = [
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {"role": "user", "content": [
                    {"type": "audio", "audio_url": None},
                    {"type": "text", "text": "Recognize all events in the audio and describe them in detail, but do not mention any time intervals or durations."},
                ]},
            ]
            text = qwen_processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
            inputs = qwen_processor(text=text, audios=[seg_data], return_tensors="pt", padding=True)
            inputs["input_ids"] = inputs["input_ids"].to(qwen_model.device)
            inputs["input_features"] = inputs["input_features"].to(qwen_model.device)
            with torch.no_grad():
                generate_ids = qwen_model.generate(**inputs, max_length=256)
                generate_ids = generate_ids[:, inputs["input_ids"].size(1):]
                event_desc = qwen_processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            event_desc = clean_description(event_desc)
            sub_result = whisper_model.transcribe(seg_file, language='en', fp16=False)
            subtitle = sub_result["text"].strip()
            outputs.append({
                "segment_index": len(outputs),
                "start_sec": round(last_end, 2),
                "end_sec": round(total_duration, 2),
                "description": event_desc,
                "subtitle": subtitle
            })
            os.remove(seg_file)

    # 输出分段明细JSON
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)
    print(f"已保存 {len(outputs)} 个事件到 {out_json}")

    # 输出summary到单独JSON
    summary_lines = []
    for seg in outputs:
        line = f"{seg['start_sec']} s - {seg['end_sec']} s, audio events: {seg['description']}. subtitle: {seg['subtitle']}"
        summary_lines.append(line)
    summary_json_obj = {
        "full_summary": summary_lines
    }
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary_json_obj, f, ensure_ascii=False, indent=2)
    print(f"已保存summary到 {summary_json}")

if __name__ == "__main__":
    process_audio(AUDIO_FILE, WORK_DIR, OUTPUT_JSON, SUMMARY_JSON)