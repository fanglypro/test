import os
import csv
from pathlib import Path
import whisper
import subprocess

# 配置路径
VIDEO_DIR = "/root/autodl-tmp/0new/omni-time/audio/merged_videos/"
OUTPUT_CSV = "/root/autodl-tmp/0new/omni-time/audio/merged_transcriptions.csv"

def extract_audio_from_video(video_path, audio_path):
    """
    从视频中提取音频
    """
    subprocess.run(["ffmpeg", "-i", video_path, "-q:a", "0", "-map", "a", audio_path, "-y"], check=True)

def transcribe_audio(audio_path, model):
    """
    使用 Whisper 对音频文件进行转录
    """
    result = model.transcribe(audio_path, fp16=False)
    segments = result["segments"]
    
    # 格式化转录结果
    transcription = []
    for segment in segments:
        start = segment["start"]
        end = segment["end"]
        text = segment["text"].strip()
        transcription.append(f"{start:.1f}-{end:.1f} seconds. {text}")
    
    return " ".join(transcription)

def main():
    # 加载 Whisper 模型
    print("加载 Whisper 模型...")
    model = whisper.load_model("base")

    # 打开 CSV 文件准备写入
    with open(OUTPUT_CSV, mode="w", encoding="utf-8", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["Video Path", "Transcription"])  # 写入表头

        # 遍历视频文件夹中的所有视频
        for video_file in Path(VIDEO_DIR).glob("*.mp4"):
            print(f"处理视频: {video_file}")
            
            # 提取音频文件
            audio_file = video_file.with_suffix(".mp3")
            extract_audio_from_video(str(video_file), str(audio_file))
            
            # 转录音频
            transcription = transcribe_audio(str(audio_file), model)
            
            # 写入到 CSV
            csvwriter.writerow([str(video_file), transcription])
            
            # 删除临时音频文件
            os.remove(audio_file)
    
    print(f"转录完成，结果已保存到: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()