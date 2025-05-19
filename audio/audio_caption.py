import os
from pathlib import Path
import whisper
import subprocess

# 路径配置
VIDEO_DIR = "/root/autodl-tmp/0new/omni-time/audio/merged_videos/"
OUTPUT_DIR = "/root/autodl-tmp/0new/omni-time/audio/audio_caption/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_audio_from_video(video_path, audio_path):
    subprocess.run([
        "ffmpeg", "-i", video_path, "-q:a", "0", "-map", "a", audio_path, "-y"
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

def transcribe_audio(audio_path, model):
    result = model.transcribe(audio_path, fp16=False)
    segments = result["segments"]
    lines = []
    for segment in segments:
        start = segment["start"]
        end = segment["end"]
        text = segment["text"].strip()
        lines.append(f"{start:.1f}-{end:.1f} seconds. {text}")
    return "\n".join(lines)

def main():
    print("加载 Whisper 模型...")
    model = whisper.load_model("base")
    for video_file in Path(VIDEO_DIR).glob("*.mp4"):
        print(f"处理视频: {video_file.name}")
        audio_file = video_file.with_suffix(".mp3")
        extract_audio_from_video(str(video_file), str(audio_file))
        transcription = transcribe_audio(str(audio_file), model)
        output_txt = os.path.join(OUTPUT_DIR, video_file.stem + ".txt")
        with open(output_txt, "w", encoding="utf-8") as f:
            f.write(transcription)
        os.remove(audio_file)
        print(f"已保存: {output_txt}")
    print("全部完成！")

if __name__ == "__main__":
    main()