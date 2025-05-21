import os
import librosa
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

model_path = "/root/autodl-tmp/model/Qwen2-Audio-7B-Instruct"
processor = AutoProcessor.from_pretrained(model_path)
model = Qwen2AudioForConditionalGeneration.from_pretrained(model_path, device_map="auto")

AUDIO_DIR = "/root/autodl-tmp/0new/omni-time/audio/part"
OUTPUT_DIR = "/root/autodl-tmp/0new/omni-time/audio/part_des"
os.makedirs(OUTPUT_DIR, exist_ok=True)

for fname in os.listdir(AUDIO_DIR):
    if fname.lower().endswith(('.mp3', '.wav', '.flac')):
        audio_file = os.path.join(AUDIO_DIR, fname)
        audio_data, _ = librosa.load(audio_file, sr=processor.feature_extractor.sampling_rate)
        audios = [audio_data]
        conversation = [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {"role": "user", "content": [
                {"type": "audio", "audio_url": None},
                {"type": "text", "text": "Please describe the audio."},
            ]},
        ]
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=text, audios=audios, return_tensors="pt", padding=True)
        inputs["input_ids"] = inputs["input_ids"].to(model.device)
        inputs["input_features"] = inputs["input_features"].to(model.device)
        generate_ids = model.generate(**inputs, max_length=256)
        generate_ids = generate_ids[:, inputs["input_ids"].size(1):]
        response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        # 保存到txt
        out_file = os.path.join(OUTPUT_DIR, os.path.splitext(fname)[0] + ".txt")
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(response)
        print(f"{fname} 的描述已保存到 {out_file}")