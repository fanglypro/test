import os
import json
import tempfile
import subprocess
import sys
import torch
import numpy as np
import soundfile as sf
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
import whisper
import re

sys.path.append("/root/autodl-tmp/0new/SHE")
from QwenAPI import QwenCallVideo

INPUT_DIR = "/root/autodl-tmp/0new/omni-time/audio/part"
OUTPUT_DIR = os.path.join(INPUT_DIR, "descriptions")
AUDIO_TMP_DIR = os.path.join(INPUT_DIR, "audio_segments_tmp")
VIDEO_TMP_DIR = os.path.join(INPUT_DIR, "video_segments_tmp")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(AUDIO_TMP_DIR, exist_ok=True)
os.makedirs(VIDEO_TMP_DIR, exist_ok=True)

# Qwen2-Audio
print("Loading Qwen2-Audio...")
qwen_model_path = "/root/autodl-tmp/model/Qwen2-Audio-7B-Instruct"
qwen_processor = AutoProcessor.from_pretrained(qwen_model_path)
qwen_model = Qwen2AudioForConditionalGeneration.from_pretrained(qwen_model_path, device_map="auto")
print("Qwen2-Audio loaded.")

# Whisper
print("Loading Whisper-large...")
whisper_model = whisper.load_model("large", device="cpu")
print("Whisper loaded.")

def clean_description(desc):
    return re.sub(r"\((?:[\d\.,\- ]+)\)", "", desc).replace("  ", " ").strip()

def get_scenes(video_path):
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=27.0))
    video_manager.set_downscale_factor()
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    scene_list = scene_manager.get_scene_list()
    return [(start.get_seconds(), end.get_seconds()) for start, end in scene_list]

def cut_video_segment(video_path, start, end, out_path):
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-ss", str(start), "-to", str(end),
        "-i", video_path, "-c", "copy", "-y", out_path
    ]
    subprocess.run(cmd, check=True)

def generate_caption_with_qwen(segment_path):
    prompt = "Please concisely and accurately describe the main visual content of this video clip in English."
    return QwenCallVideo(prompt, segment_path)

def process_video(video_path):
    scenes = get_scenes(video_path)
    video_results = []
    for idx, (start, end) in enumerate(scenes):
        with tempfile.NamedTemporaryFile(suffix=".mp4", dir=VIDEO_TMP_DIR, delete=False) as tmp_vid:
            cut_video_segment(video_path, start, end, tmp_vid.name)
            try:
                desc = generate_caption_with_qwen(tmp_vid.name)
                desc = desc.strip().replace("\n", " ")
            except Exception as e:
                desc = f"[Qwen call failed: {e}]"
            finally:
                os.remove(tmp_vid.name)
        video_results.append({
            "scene_index": idx,
            "start_sec": round(start, 2),
            "end_sec": round(end, 2),
            "description": desc
        })
    return video_results

def process_audio(audio_path):
    result = whisper_model.transcribe(audio_path, language='en', fp16=False)
    segments = result["segments"]

    y, sr = sf.read(audio_path)
    if len(y.shape) > 1:
        y = y[:, 0]

    outputs = []
    last_end = 0
    for idx, seg in enumerate(segments):
        start_sec = seg["start"]
        end_sec = seg["end"]
        start = int(start_sec * sr)
        end = int(end_sec * sr)
        if end <= start:
            continue
        audio_seg = y[start:end]
        seg_file = os.path.join(AUDIO_TMP_DIR, f"segment_{idx}.wav")
        sf.write(seg_file, audio_seg, sr)

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
            "segment_index": idx,
            "start_sec": round(start_sec, 2),
            "end_sec": round(end_sec, 2),
            "description": event_desc,
            "subtitle": subtitle
        })
        os.remove(seg_file)
        last_end = end_sec

    # Tail segment for missed audio at the end
    total_duration = len(y) / sr
    if last_end < total_duration - 0.1:
        start = int(last_end * sr)
        end = len(y)
        audio_seg = y[start:end]
        seg_file = os.path.join(AUDIO_TMP_DIR, "segment_tail.wav")
        sf.write(seg_file, audio_seg, sr)
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

    # summary
    summary_lines = []
    for seg in outputs:
        line = f"{seg['start_sec']} s - {seg['end_sec']} s, audio events: {seg['description']}. subtitle: {seg['subtitle']}"
        summary_lines.append(line)
    summary_json_obj = {
        "full_summary": summary_lines
    }
    return outputs, summary_json_obj

def match_audio_to_video(video_path):
    # Try to find matching audio file for a given video (same basename, different ext)
    video_base = os.path.splitext(os.path.basename(video_path))[0]
    for ext in [".wav", ".mp3", ".aac", ".flac", ".m4a"]:
        audio_path = os.path.join(INPUT_DIR, video_base + ext)
        if os.path.isfile(audio_path):
            return audio_path
    return None

def main():
    files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv"))]
    for vfile in files:
        video_path = os.path.join(INPUT_DIR, vfile)
        print(f"\nProcessing video: {video_path}")

        # video description
        video_results = process_video(video_path)
        video_desc_out = os.path.join(OUTPUT_DIR, vfile + ".video_description.json")
        with open(video_desc_out, "w", encoding="utf-8") as f:
            json.dump(video_results, f, ensure_ascii=False, indent=2)
        print(f"Saved video description to {video_desc_out}")

        # audio description
        audio_path = match_audio_to_video(video_path)
        if audio_path is None:
            # fallback: extract audio from video
            audio_path = os.path.join(AUDIO_TMP_DIR, os.path.splitext(vfile)[0] + "_audio_tmp.wav")
            cmd = [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", "-y", audio_path
            ]
            subprocess.run(cmd, check=True)
            remove_audio_after = True
        else:
            remove_audio_after = False

        audio_segments, audio_summary = process_audio(audio_path)
        audio_seg_out = os.path.join(OUTPUT_DIR, vfile + ".audio_segments.json")
        audio_sum_out = os.path.join(OUTPUT_DIR, vfile + ".audio_summary.json")
        with open(audio_seg_out, "w", encoding="utf-8") as f:
            json.dump(audio_segments, f, ensure_ascii=False, indent=2)
        with open(audio_sum_out, "w", encoding="utf-8") as f:
            json.dump(audio_summary, f, ensure_ascii=False, indent=2)
        print(f"Saved audio segments to {audio_seg_out}")
        print(f"Saved audio summary to {audio_sum_out}")
        if remove_audio_after:
            os.remove(audio_path)

if __name__ == "__main__":
    main()