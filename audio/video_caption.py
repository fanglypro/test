import os
import cv2
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

VIDEO_DIR = "/root/autodl-tmp/0new/omni-time/audio/merged_videos"
OUTPUT_DIR = "/root/autodl-tmp/0new/omni-time/audio/video_captions"

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

def get_scenes(video_path):
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=27.0))
    video_manager.set_downscale_factor()
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    scene_list = scene_manager.get_scene_list()
    return [(start.get_seconds(), end.get_seconds()) for start, end in scene_list]

def extract_middle_frame(video_path, start_sec, end_sec):
    cap = cv2.VideoCapture(video_path)
    mid_sec = (start_sec + end_sec) / 2
    cap.set(cv2.CAP_PROP_POS_MSEC, mid_sec * 1000)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

def generate_caption(img):
    inputs = processor(images=img, return_tensors="pt").to(device)
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption.strip().capitalize()

def process_video(video_path, output_txt):
    scenes = get_scenes(video_path)
    results = []
    for start, end in scenes:
        img = extract_middle_frame(video_path, start, end)
        if img is None:
            continue
        caption = generate_caption(img)
        results.append(f'{start:.1f}-{end:.1f} seconds: "{caption}."')
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(results))
    print(f"完成: {output_txt}")

def main():
    for filename in os.listdir(VIDEO_DIR):
        if not filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            continue
        video_path = os.path.join(VIDEO_DIR, filename)
        output_txt = os.path.join(OUTPUT_DIR, os.path.splitext(filename)[0] + ".txt")
        process_video(video_path, output_txt)

if __name__ == "__main__":
    main()