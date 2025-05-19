import os
import json
import subprocess
from pathlib import Path
from collections import defaultdict

# 配置路径
VIDEO_PATHS_JSON = "/root/autodl-tmp/0new/AVSBench/SVA/video_paths.json"
OUTPUT_VIDEO_DIR = "/root/autodl-tmp/0new/omni-time/audio/merged_videos/"
OUTPUT_AUDIO_DIR = "/root/autodl-tmp/0new/omni-time/audio/merged_audios/"

# 确保输出目录存在
Path(OUTPUT_VIDEO_DIR).mkdir(parents=True, exist_ok=True)
Path(OUTPUT_AUDIO_DIR).mkdir(parents=True, exist_ok=True)

def parse_video_paths(video_paths_json):
    """
    解析 JSON 文件，按 `diaXX` 标签分组视频文件
    """
    with open(video_paths_json, "r") as f:
        video_paths = json.load(f)
    
    grouped_videos = defaultdict(list)
    for video_path in video_paths:
        # 提取 dia 标签，例如 "dia97", "dia27"
        dialogue_label = Path(video_path).stem.split("_")[0]  # 提取 "dia97" 等
        grouped_videos[dialogue_label].append(video_path)
    
    return grouped_videos

def merge_videos(video_paths, output_video_path):
    """
    使用 ffmpeg 合并多个视频文件
    """
    # 创建一个临时文件，保存所有视频路径
    with open("temp_video_list.txt", "w") as f:
        for video_path in video_paths:
            f.write(f"file '{video_path}'\n")
    
    # 调用 ffmpeg 合并视频
    subprocess.run([
        "ffmpeg", "-f", "concat", "-safe", "0", "-i", "temp_video_list.txt", 
        "-c", "copy", output_video_path, "-y"
    ], check=True)
    print(f"视频合并完成: {output_video_path}")
    
    # 删除临时文件
    os.remove("temp_video_list.txt")

def merge_audios(video_paths, output_audio_path):
    """
    使用 ffmpeg 提取音频并合并
    """
    temp_audio_paths = []
    # 提取视频中的音频
    for video_path in video_paths:
        audio_path = f"{Path(video_path).stem}.mp3"
        temp_audio_paths.append(audio_path)
        subprocess.run([
            "ffmpeg", "-i", video_path, "-q:a", "0", "-map", "a", audio_path, "-y"
        ], check=True)
    
    # 创建一个临时文件，保存所有音频路径
    with open("temp_audio_list.txt", "w") as f:
        for audio_path in temp_audio_paths:
            f.write(f"file '{audio_path}'\n")
    
    # 调用 ffmpeg 合并音频
    subprocess.run([
        "ffmpeg", "-f", "concat", "-safe", "0", "-i", "temp_audio_list.txt", 
        "-c", "copy", output_audio_path, "-y"
    ], check=True)
    print(f"音频合并完成: {output_audio_path}")
    
    # 删除临时文件和中间音频文件
    os.remove("temp_audio_list.txt")
    for temp_audio_path in temp_audio_paths:
        os.remove(temp_audio_path)

def main():
    # 解析 JSON 文件
    grouped_videos = parse_video_paths(VIDEO_PATHS_JSON)
    
    for dialogue_label, video_paths in grouped_videos.items():
        print(f"处理对话标签: {dialogue_label}")
        
        # 合并视频
        output_video_path = os.path.join(OUTPUT_VIDEO_DIR, f"{dialogue_label}_merged.mp4")
        merge_videos(video_paths, output_video_path)
        
        # 合并音频
        output_audio_path = os.path.join(OUTPUT_AUDIO_DIR, f"{dialogue_label}_merged.mp3")
        merge_audios(video_paths, output_audio_path)

if __name__ == "__main__":
    main()