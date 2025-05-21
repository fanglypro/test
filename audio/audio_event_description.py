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
OUTPUT_JSON = os.path.join(WORK_DIR, "VM_segments_mfcc.json")

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

# ================== MFCC分割 ==================
def mfcc_segment(audio_file, sr=16000, min_len=2.0, hop_length=2048):
    y, sr = librosa.load(audio_file, sr=sr)
    total_sec = len(y) / sr
    print(f"音频总长度: {total_sec:.2f} 秒")
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length)
    delta = np.mean(np.abs(np.diff(mfcc, axis=1)), axis=0)
    threshold = float(delta.mean() + delta.std())
    print(f"MFCC delta: max={delta.max():.2f}, min={delta.min():.2f}, mean={delta.mean():.2f}, std={delta.std():.2f}")
    print(f"自动设定分割阈值: {threshold:.2f}")

    # 检测边界
    boundaries = [0]
    for i, v in enumerate(delta):
        if v > threshold:
            idx = i + 1
            # 相邻分割点间隔要大于min_len一半
            if idx - boundaries[-1] > int(min_len * sr / hop_length / 2):
                boundaries.append(idx)
    boundaries.append(mfcc.shape[1])
    boundaries = sorted(set(boundaries))

    # 生成初步片段 (start_sample, end_sample)
    segments = []
    for i in range(len(boundaries) - 1):
        start_idx = boundaries[i] * hop_length
        end_idx = min(boundaries[i + 1] * hop_length, len(y))
        if (end_idx - start_idx) / sr >= min_len / 2:  # 只要大于最小长度一半先保留
            segments.append((int(start_idx), int(end_idx)))
    print(f"MFCC分割得到 {len(segments)} 个片段")
    return y, sr, segments

# ================== CLAP嵌入 ==================
def get_clap_embedding(audio_path, model, processor, device="cpu"):
    # CLAP transformers 需要48000采样率
    audio, _ = librosa.load(audio_path, sr=48000)
    inputs = processor(audios=audio, return_tensors="pt", sampling_rate=48000)
    with torch.no_grad():
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model.get_audio_features(**inputs)
    embedding = outputs.squeeze()  # [emb_dim]
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
    y, sr, segments = mfcc_segment(audio_file)
    if not segments:
        print("未检测到有效片段，请调整分割参数！")
        return

    # 保存所有初步分割片段
    segment_paths = []
    for idx, (start, end) in enumerate(segments):
        seg = y[start:end]
        seg_file = os.path.join(work_dir, f"tmp_segment_{idx}.wav")
        sf.write(seg_file, seg, sr)
        segment_paths.append(seg_file)

    # CLAP语义聚合
    merged_indices = merge_by_semantics(segment_paths, clap_model, clap_processor, similarity_thres=0.88, device=device)
    print(f"CLAP聚合后片段数: {len(merged_indices)}")

    results = []
    for seg_idx, (start_i, end_i) in enumerate(merged_indices):
        start_sample = segments[start_i][0]
        end_sample = segments[end_i][1]
        seg = y[start_sample:end_sample]
        seg_file = os.path.join(work_dir, f"tmp_final_segment_{seg_idx}.wav")
        sf.write(seg_file, seg, sr)

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
        # Whisper生成字幕
        try:
            asr = whisper_model.transcribe(seg_file, language='en', fp16=False)
            subtitle = asr["text"].strip()
        except Exception as e:
            subtitle = ""
        result = {
            "segment_index": seg_idx,
            "start_sec": round(start_sample / sr, 2),
            "end_sec": round(end_sample / sr, 2),
            "description": desc,
            "subtitle": subtitle
        }
        results.append(result)
        print(f"片段{seg_idx} [{result['start_sec']}s - {result['end_sec']}s]")
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