import os
from pathlib import Path
import whisper

# 路径配置
AUDIO_DIR = "/root/autodl-tmp/0new/omni-time/audio/part/"
OUTPUT_DIR = "/root/autodl-tmp/0new/omni-time/audio/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
    for audio_file in Path(AUDIO_DIR).glob("*"):
        if audio_file.suffix.lower() not in [".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg"]:
            continue
        print(f"处理音频: {audio_file.name}")
        transcription = transcribe_audio(str(audio_file), model)
        output_txt = os.path.join(OUTPUT_DIR, audio_file.stem + ".txt")
        with open(output_txt, "w", encoding="utf-8") as f:
            f.write(transcription)
        print(f"已保存: {output_txt}")
    print("全部完成！")

if __name__ == "__main__":
    main()