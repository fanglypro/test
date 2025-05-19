import os
import sys
import csv
import tempfile
import subprocess
from pathlib import Path

# 配置依赖路径
sys.path.append("/root/autodl-tmp/0new/SHE")
from QwenAPI import QwenCallVideo
import whisper
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector

# 目录配置（可根据需要修改，避免与旧目录重复）
VIDEO_DIR = "/root/autodl-tmp/0new/omni-time/audio/part"
QA_VIDEO_CAPTION_DIR = "/root/autodl-tmp/0new/omni-time/audio/qa_video_caption"
QA_AUDIO_CAPTION_DIR = "/root/autodl-tmp/0new/omni-time/audio/qa_audio_caption"
CSV_OUT = "video_qa_output.csv"
os.makedirs(QA_VIDEO_CAPTION_DIR, exist_ok=True)
os.makedirs(QA_AUDIO_CAPTION_DIR, exist_ok=True)

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

def generate_video_caption(video_path):
    scenes = get_scenes(video_path)
    result_lines = []
    for idx, (start, end) in enumerate(scenes):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_vid:
            cut_video_segment(video_path, start, end, tmp_vid.name)
            try:
                prompt = "Please concisely and accurately describe the main visual content of this video clip in English."
                desc = QwenCallVideo(prompt, tmp_vid.name)
                desc = desc.strip().replace("\n", " ")
            except Exception as e:
                desc = f"[Qwen调用失败: {e}]"
            finally:
                os.remove(tmp_vid.name)
        result_lines.append(f"{start:.1f}-{end:.1f} seconds: \"{desc}\"")
    return "\n".join(result_lines)

def extract_audio_from_video(video_path, audio_path):
    subprocess.run([
        "ffmpeg", "-i", video_path, "-q:a", "0", "-map", "a", audio_path, "-y"
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

def generate_audio_caption(video_path, model):
    audio_path = Path(video_path).with_suffix(".mp3")
    extract_audio_from_video(str(video_path), str(audio_path))
    result = model.transcribe(str(audio_path), fp16=False)
    segments = result["segments"]
    lines = []
    for segment in segments:
        start = segment["start"]
        end = segment["end"]
        text = segment["text"].strip()
        lines.append(f"{start:.1f}-{end:.1f} seconds. {text}")
    os.remove(audio_path)
    return "\n".join(lines)

def generate_qa(video_desc, audio_desc, video_path):
    prompt = f"""
You are a helpful assistant designed to output JSON.

I will give you descriptions of audio and video segments, each with timestamps. Your task is to generate a single question-answer pair that tests a student's ability to understand and integrate the information from both modalities.

Guidelines:

1. Generate exactly one concrete and specific question that requires combining information from more than one provided description.
   - Avoid compound or multi-part questions.
   - Do not include timestamps in the question.
   - Avoid general or vague phrasing. Make the question specific and focused.

2. The answer should be concise and directly supported by both the audio and/or video descriptions.

3. If multiple descriptions refer to the same or continuous actions, merge them and refer to the merged timespans in the evidence.

4. If the input includes descriptions of similar actions (e.g., using different hands), ask about their time spans directly, and provide only the relevant tagged spans in the answer.

5. Include an "Evidence" section:
   - Explain how the answer is supported.
   - Reference the relevant time spans from both audio and video.
   - Clearly describe how the audio and video information work together to support the answer.
   - Include the timestamps in the text, not in parentheses or as inline notes.

Your output must be a JSON object in the following format:
{{
  "Question": "...",
  "Answer": "...",
  "Evidence": "..."
}}
---

Example 1:

Input:
Audio:  
0.0–7.3 seconds. Well, I went over to Kyle's last night to pick up a few things and we got to reminiscing.  
7.3–8.3 seconds. I'm sorry.  
8.3–10.3 seconds. Ah.  
10.3–13.7 seconds. Feibes, you were right about her.  
13.7–22.0 seconds. We talked through most of the night and we realized that the reason we're so angry with each other is because there are still feelings there.

Video:  
0.0–7.2 seconds: A woman sitting next to a man in a restaurant.  
7.2–9.9 seconds: The big bang on the big bang.  
9.9–13.5 seconds: A man in a suit and tie walking through a bar.  
13.5–15.3 seconds: A man in a suit and tie sitting at a table.  
15.3–17.9 seconds: A woman sitting in a chair with a cup.  
17.9–22.1 seconds: A man and woman sitting at a table.

Output:
{{
  "Question": "What realization do the speakers reach about their relationship after their conversation at the restaurant?",
  "Answer": "They realize that their anger toward each other is due to lingering feelings between them.",
  "Evidence": "In the audio from 13.7 to 22.0 seconds, it is stated that 'we realized that the reason we're so angry with each other is because there are still feelings there.' The video from 0.0 to 22.1 seconds shows the man and woman spending time together at a restaurant, which supports the idea that they are reconnecting emotionally."
}}

---

Example 2:

Input:
Audio:  
0.0–6.5 seconds. And Carol and I'd be out and she'd see some beautiful woman and she'd be  
6.5–8.8 seconds. Ross, you know, look at her.  
8.8–14.1 seconds. And I'd think, I mean, we've been together seven years.  
14.1–17.1 seconds. She's the only woman that's ever loved me.  
17.1–19.6 seconds. She's the only woman I've ever...  
19.6–24.5 seconds. My marriage, I think my marriage is kind of over.  
24.5–26.4 seconds. Oh no, why?  
26.4–29.4 seconds. Oh god!  
29.4–32.0 seconds. I don't believe it!  
32.0–35.8 seconds. Oh, you're poor funny.  
35.8–39.2 seconds. Hey, do you think that Susan person is her lover?  
39.2–44.6 seconds. Because Carol's a lesbian.  
44.6–51.3 seconds. And I'm not one.  
51.3–54.1 seconds. And apparently it's not a mix and match situation.

Video:  
0.0–10.2 seconds: A man and woman standing at a bar.  
10.2–13.5 seconds: A man and woman playing pool in a bar.  
13.5–19.5 seconds: A man in a plaid shirt standing in front of a bar.  
19.5–24.1 seconds: A man in a plaid shirt is walking down the street.  
24.1–26.3 seconds: A woman in a white dress standing in front of a bar.  
26.3–33.8 seconds: A woman sitting at a bar with a man standing behind her.  
33.8–38.2 seconds: A woman sitting at a table with a drink.  
38.2–38.9 seconds: A man sitting at a bar with a drink.  
38.9–41.1 seconds: A man in a plaid shirt standing in front of a bar.  
41.1–43.0 seconds: A woman in a white dress sitting at a table.  
43.0–48.1 seconds: A man in a plaid shirt standing in front of a bar.  
48.1–50.3 seconds: A woman in a white dress standing in front of a pink wall.  
50.3–54.1 seconds: A man in a plaid shirt standing in front of a bar.

Output:
{{
  "Question": "What causes Ross to believe that his marriage is ending?",
  "Answer": "Ross believes his marriage is ending because his wife Carol is a lesbian.",
  "Evidence": "In the audio from 19.6 to 24.5 seconds, Ross says 'my marriage is kind of over,' and from 39.2 to 44.6 seconds, he explains 'because Carol's a lesbian.' The video throughout these moments shows Ross and Carol in different settings and never interacting directly, reinforcing the idea of emotional and physical separation."
}}

---

Now, for the following transcript, generate a Question, Answer, and Evidence as above:

Input:
Audio:
{audio_desc}

Video:
{video_desc}
"""
    try:
        qa = QwenCallVideo(prompt, video_path)
        return qa.strip().replace("\n", " ")
    except Exception as e:
        return f"[Qwen调用失败: {e}]"

def main():
    model = whisper.load_model("base")
    rows = []
    for fname in os.listdir(VIDEO_DIR):
        if not fname.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
            continue
        video_path = os.path.join(VIDEO_DIR, fname)
        print(f"\n处理视频: {video_path}")

        # 生成视频描述
        video_caption_txt = os.path.join(QA_VIDEO_CAPTION_DIR, Path(video_path).stem + ".txt")
        if not os.path.exists(video_caption_txt):
            video_desc = generate_video_caption(video_path)
            with open(video_caption_txt, "w", encoding="utf-8") as f:
                f.write(video_desc)
        else:
            with open(video_caption_txt, "r", encoding="utf-8") as f:
                video_desc = f.read().strip()

        # 生成音频描述
        audio_caption_txt = os.path.join(QA_AUDIO_CAPTION_DIR, Path(video_path).stem + ".txt")
        if not os.path.exists(audio_caption_txt):
            audio_desc = generate_audio_caption(video_path, model)
            with open(audio_caption_txt, "w", encoding="utf-8") as f:
                f.write(audio_desc)
        else:
            with open(audio_caption_txt, "r", encoding="utf-8") as f:
                audio_desc = f.read().strip()

        # 生成问答
        qa = generate_qa(video_desc, audio_desc, video_path)

        rows.append([
            video_path,
            video_desc,
            audio_desc,
            qa
        ])

    # 写csv
    with open(CSV_OUT, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["视频路径", "视频描述", "音频描述", "问答"])
        writer.writerows(rows)
    print(f"\n全部完成！结果已保存到: {CSV_OUT}")

if __name__ == "__main__":
    main()