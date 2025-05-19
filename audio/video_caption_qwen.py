import os
import sys
import tempfile
import subprocess
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector

sys.path.append("/root/autodl-tmp/0new/SHE")
from QwenAPI import QwenCallVideo

VIDEO_DIR = "/root/autodl-tmp/0new/omni-time/audio/merged_videos"
OUTPUT_DIR = "/root/autodl-tmp/0new/omni-time/audio/video_caption_qwen"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
    prompt = "请Please concisely and accurately describe the main visual content of this video clip in English."
    return QwenCallVideo(prompt, segment_path)

def process_one_video(video_path):
    scenes = get_scenes(video_path)
    result_lines = []
    for idx, (start, end) in enumerate(scenes):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_vid:
            cut_video_segment(video_path, start, end, tmp_vid.name)
            try:
                desc = generate_caption_with_qwen(tmp_vid.name)
                desc = desc.strip().replace("\n", " ")
            except Exception as e:
                desc = f"[Qwen调用失败: {e}]"
            finally:
                os.remove(tmp_vid.name)
        result_lines.append(f"{start:.1f}-{end:.1f} seconds: \"{desc}\"")
    out_txt = os.path.join(
        OUTPUT_DIR, os.path.splitext(os.path.basename(video_path))[0] + ".txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(result_lines))
    print(f"已完成 {out_txt}")

def main():
    for fname in os.listdir(VIDEO_DIR):
        if not fname.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
            continue
        vpath = os.path.join(VIDEO_DIR, fname)
        process_one_video(vpath)

if __name__ == "__main__":
    main()