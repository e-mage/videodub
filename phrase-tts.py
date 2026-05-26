# -*- coding: utf-8 -*-
"""
phrase-tts.py

Batch speech generation using the Qwen3-TTS/Base model for voice cloning.
"""

import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel
import argparse
import signal
import platform
import json
import librosa
import os

def main():
    parser = argparse.ArgumentParser(description="Batch speech generation using the Qwen3-TTS/Base model for voice cloning.")
    parser.add_argument("--tasks_file", type=str, required=True, help="Path to the JSON file containing TTS tasks. Tasks file example: { \"tts_tasks\": [ { \"text\": \"Давай же, пойдём.\", \"output_file\": \"/path/to/tts.wav\", \"ref_audio\": \"/path/to/ref.wav\", \"source_duration\": 0.74, \"ref_text\": \"Come on, let's go.\" }, {...}, {...} ] }")
    parser.add_argument("--language", type=str, default="Russian", help="Language of synthesized text. Default = 'Russian'. The next languages are supported by TTS engine: Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, and Italian.")
    args = parser.parse_args()

    # Capturing randomness for reproducibility
    """
    SEED = 42
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
    """

    # Load the TTS model
    clone_model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    # Read JSON file with TTS tasks
    with open(args.tasks_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    tts_tasks = data['tts_tasks']

    # Process each TTS task (from args.tasks_file)
    for task in tts_tasks:
        #speaker_id = task['speaker_id']
        text_to_synthesize = task['text']
        output_file = task['output_file']
        ref_audio = task['ref_audio']
        ref_text = task['ref_text']
        ref_duration = task['source_duration']

        print(f"\n{'#'*20}\n📄 Creating voice clone prompt from:\n    ref_audio: {ref_audio},\n    ref_text: {ref_text}\n    ...")
        try:
            voice_clone_prompt = clone_model.create_voice_clone_prompt(
                ref_audio=ref_audio,
                ref_text=ref_text,
            )
            voice_clone_prompt_fallback = clone_model.create_voice_clone_prompt(
                ref_audio=ref_audio,
                ref_text=ref_text,
                x_vector_only_mode=True,
            )

        except Exception as e:
            print(f"❌ ERROR: Failed to create voice clone prompt of ref_audio: {ref_audio} and ref_text: {ref_text}. Error: {e}")
            continue

        """
        Make several attempts to synthesize speech until an acceptable phrase length is achieved.
        """
        max_attempts = 16
        fallback = False
        min_dur_tts_audio = None # Audio tts with minimal duration among attempts
        min_dur = -1.0 # Minimal duration of audio tts (in seconds)
        min_dur_sr = None
        for i in range(max_attempts):
            print(f"📄 Generating speech for:\n    text: '{text_to_synthesize}'\n    (Attempt {i+1}/{max_attempts}){' (Fallback !!!)' if fallback else ''}")

            timeout_seconds = 10 + len(text_to_synthesize) // 4
            if platform.system() != "Windows":
                class TimeoutError(Exception): pass
                def handler(signum, frame): raise TimeoutError()
                signal.signal(signal.SIGALRM, handler)
                signal.alarm(timeout_seconds)

            try:
                wavs, sr = clone_model.generate_voice_clone(
                    text=text_to_synthesize,
                    language=args.language,
                    voice_clone_prompt=voice_clone_prompt_fallback if fallback else voice_clone_prompt,
                    top_p=0.9, #default =1.0
                    temperature=0.8, # default =0.9
                    subtalker_top_p=0.9,
                    subtalker_temperature=0.8,
                    #max_new_tokens=2048, #default = 2048
                    #repetition_penalty=1.1, #default = 1.05
                    #seed=SEED,
                )
            except TimeoutError:
                print(f"Speech generation timed out after {timeout_seconds} seconds. Retrying...")
                if platform.system() != "Windows":
                    signal.alarm(0)
                continue
            finally:
                if platform.system() != "Windows":
                    signal.alarm(0)

            print(f"📄 Triming generated audio...")
            y_trimmed, _ = librosa.effects.trim(wavs[0], top_db=30)
            # Report results after trimming
            original_dur = len(wavs[0]) / sr
            trimmed_dur = len(y_trimmed) / sr
            print(f"   Original: {original_dur:.2f}s | Trimmed: {trimmed_dur:.2f}s")
            # Remembering the shortest option
            if trimmed_dur < min_dur or min_dur < 0.0:
                min_dur = trimmed_dur
                min_dur_tts_audio = y_trimmed
                min_dur_sr = sr

            try:
                duration_ratio_threshold = 2.3
                if i > max_attempts/4:
                    duration_ratio_threshold += 0.5
                if i > max_attempts*2/4:
                    duration_ratio_threshold += 1.0
                if i > max_attempts*3/4:
                    #duration_ratio_threshold += 1.0
                    fallback = True
                if trimmed_dur <= duration_ratio_threshold * ref_duration:
                    print("Generated audio duration is acceptable.")
                    print(f"Saving generated and trimed audio to {output_file}...")
                    sf.write(output_file, y_trimmed, sr)
                    break
                elif min_dur <= duration_ratio_threshold * ref_duration:
                    print(f"Saving MINIMAL LENGTH audio to {output_file}...")
                    sf.write(output_file, min_dur_tts_audio, min_dur_sr)
                    break
                else:
                    print(f"⚠️ Warning: Generated audio is {trimmed_dur / ref_duration:.2f} times longer than the reference audio (threshold: {duration_ratio_threshold:.1f}x). Retrying...")
            except Exception as e:
                print(f"❌ Error getting audio duration: {e}")
                break
        else:
            print(f"❌ Failed to generate an audio of acceptable length for '{text_to_synthesize}' after {max_attempts} attempts.")
            if os.path.exists(output_file):
                print(f"Deleting file with not acceptable length: {output_file}...")
                os.remove(output_file)

    print("\n✅ TTS batch is done.\n")

if __name__ == '__main__':
    main()
