import os
import json
import librosa
import numpy as np
import torch
import soundfile as sf
from transformers import ClapModel, ClapProcessor
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
import whisper

AUDIO_FILE = "/root/autodl-tmp/0new/omni-time/audio/part/VM.mp3"
WORK_DIR = "/root/autodl-tmp/0new/omni-time/audio"
OUTPUT_JSON = os.path.join(WORK_DIR, "VM_new_segments.json")
SEGMENT_DIR = os.path.join(WORK_DIR, "whisper_segments")
os.makedirs(SEGMENT_DIR, exist_ok=True)

# ================== 加载模型 ==================
print("正在加载 transformers 版 CLAP...")
clap_model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
clap_processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
clap_model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
clap_model = clap_model.to(device)
print("CLAP加载完成.")

print("正在加载Qwen2-Audio-7B-Instruct...")
model_path = "/root/autodl-tmp/model/Qwen2-Audio-7B-Instruct"
qwen_processor = AutoProcessor.from_pretrained(model_path)
qwen_model = Qwen2AudioForConditionalGeneration.from_pretrained(model_path, device_map="auto")
print("Qwen2-Audio模型加载完成.")

print("正在加载Whisper-large到CPU...")
whisper_model = whisper.load_model("large", device="cpu")
print("Whisper模型加载完成.")

# ================== CLAP嵌入 ==================
def get_clap_embedding(audio_path, model, processor, device="cpu"):
    audio, _ = librosa.load(audio_path, sr=48000)
    inputs = processor(audios=audio, return_tensors="pt", sampling_rate=48000)
    with torch.no_grad():
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model.get_audio_features(**inputs)
    embedding = outputs.squeeze()
    return embedding.cpu()

def merge_by_semantics(segment_paths, clap_model, clap_processor, similarity_thres=0.88, device="cpu"):
    embeddings = [get_clap_embedding(p, clap_model, clap_processor, device=device) for p in segment_paths]
    embeddings = [e / e.norm() for e in embeddings]
    merged = []
    i = 0
    while i < len(segment_paths):
        start = i
        end = i
        while end + 1 < len(segment_paths):
            sim = torch.cosine_similarity(embeddings[end], embeddings[end + 1], dim=0).item()
            if sim > similarity_thres:
                end += 1
            else:
                break
        merged.append((start, end))
        i = end + 1
    return merged

# ================== 主流程 ==================
def process_audio(audio_file, work_dir, out_json, device="cpu"):
    # 1. Whisper分段
    y, sr = librosa.load(audio_file, sr=16000)
    tmp_wav = os.path.join(work_dir, "tmp_for_whisper.wav")
    sf.write(tmp_wav, y, sr)
    result = whisper_model.transcribe(tmp_wav, language='en', fp16=False)
    segments = result["segments"]

    # 2. 保存whisper分段音频
    segment_paths = []
    segment_times = []
    for idx, seg in enumerate(segments):
        start = int(seg["start"] * sr)
        end = int(seg["end"] * sr)
        audio_seg = y[start:end]
        seg_file = os.path.join(SEGMENT_DIR, f"segment_{idx}.wav")
        sf.write(seg_file, audio_seg, sr)
        segment_paths.append(seg_file)
        segment_times.append((seg["start"], seg["end"]))
    os.remove(tmp_wav)

    # 3. CLAP语义聚合（可选，如果你希望自动合并极其相似的连续段落，否则可以直接for seg in segments循环输出）
    merged_indices = merge_by_semantics(segment_paths, clap_model, clap_processor, similarity_thres=0.88, device=device)
    print(f"CLAP聚合后片段数: {len(merged_indices)}")

    results = []
    for seg_idx, (start_i, end_i) in enumerate(merged_indices):
        start_sec = segment_times[start_i][0]
        end_sec = segment_times[end_i][1]
        # 合并音频
        audio_merge = np.concatenate([librosa.load(segment_paths[i], sr=sr)[0] for i in range(start_i, end_i+1)])
        seg_file = os.path.join(SEGMENT_DIR, f"final_segment_{seg_idx}.wav")
        sf.write(seg_file, audio_merge, sr)

        # Qwen2-Audio生成描述
        conversation = [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {"role": "user", "content": [
                {"type": "audio", "audio_url": None},
                {"type": "text", "text": "Please describe the audio."},
            ]},
        ]
        text = qwen_processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        seg_data, _ = librosa.load(seg_file, sr=qwen_processor.feature_extractor.sampling_rate)
        inputs = qwen_processor(text=text, audios=[seg_data], return_tensors="pt", padding=True)
        inputs["input_ids"] = inputs["input_ids"].to(qwen_model.device)
        inputs["input_features"] = inputs["input_features"].to(qwen_model.device)
        with torch.no_grad():
            generate_ids = qwen_model.generate(**inputs, max_length=256)
            generate_ids = generate_ids[:, inputs["input_ids"].size(1):]
            desc = qwen_processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        # Whisper生成字幕（直接拼接合并段的segments文本更快，也可以重新识别）
        subtitle = " ".join([segments[i]["text"].strip() for i in range(start_i, end_i+1)])

        result = {
            "segment_index": seg_idx,
            "start_sec": round(start_sec, 2),
            "end_sec": round(end_sec, 2),
            "description": desc,
            "subtitle": subtitle
        }
        results.append(result)
        print(f"片段{seg_idx} [{start_sec:.2f}s - {end_sec:.2f}s]")
        print("描述:", desc)
        print("字幕:", subtitle)
        print("="*40)
        os.remove(seg_file)
    # 清理临时片段
    for p in segment_paths:
        if os.path.exists(p):
            os.remove(p)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"已保存 {len(results)} 个最终片段到 {out_json}")

if __name__ == "__main__":
    process_audio(AUDIO_FILE, WORK_DIR, OUTPUT_JSON, device=device)