import os
import json
import subprocess
from pathlib import Path
import whisper

# 配置路径
VIDEO_PATHS_JSON = "/root/autodl-tmp/0new/AVSBench/SVA/video_paths.json"
AUDIO_DIR = "/root/autodl-tmp/0new/omni-time/audio/audio_files/"
TRANSCRIPTIONS_DIR = "/root/autodl-tmp/0new/omni-time/audio/transcriptions/"

# 确保输出目录存在
Path(AUDIO_DIR).mkdir(parents=True, exist_ok=True)
Path(TRANSCRIPTIONS_DIR).mkdir(parents=True, exist_ok=True)

def extract_audio_from_videos(video_paths, audio_dir):
    """
    使用 ffmpeg 从视频中提取音频
    """
    for video_path in video_paths:
        audio_path = os.path.join(audio_dir, f"{Path(video_path).stem}.mp3")
        if not os.path.exists(audio_path):  # 如果音频文件已存在，则跳过
            print(f"提取音频: {video_path} -> {audio_path}")
            subprocess.run(["ffmpeg", "-i", video_path, "-q:a", "0", "-map", "a", audio_path, "-y"], check=True)
        else:
            print(f"音频已存在，跳过: {audio_path}")

def transcribe_audio_files(audio_dir, transcriptions_dir, model_name="base"):
    """
    使用 Whisper 对音频文件进行转录
    """
    model = whisper.load_model(model_name)
    for audio_file in Path(audio_dir).glob("*.mp3"):
        print(f"转录音频: {audio_file}")
        result = model.transcribe(str(audio_file), fp16=False)
        segments = result["segments"]

        # 格式化转录结果
        transcription = []
        for segment in segments:
            start = segment["start"]
            end = segment["end"]
            text = segment["text"].strip()
            transcription.append(f"{start:.1f}-{end:.1f} seconds. {text}")

        # 保存转录结果
        transcription_file = Path(transcriptions_dir) / f"{audio_file.stem}_transcription.txt"
        with open(transcription_file, "w", encoding="utf-8") as f:
            f.write("Transcribed speech: " + " ".join(transcription))
        print(f"转录结果已保存: {transcription_file}")

def main():
    # 读取视频路径
    with open(VIDEO_PATHS_JSON, "r") as f:
        video_paths = json.load(f)

    # 提取音频
    extract_audio_from_videos(video_paths, AUDIO_DIR)

    # 转录音频
    transcribe_audio_files(AUDIO_DIR, TRANSCRIPTIONS_DIR)

if __name__ == "__main__":
    main()