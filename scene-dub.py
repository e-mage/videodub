# -*- coding: utf-8 -*-
"""
scene-dub.py

Transcribes and optionally translates an audio or video file, then splits it into chunks.
Optionally run TTS with external script 'phrase-tts.py'
"""

import argparse
import os
import sys
import subprocess
import time
import requests
import json
import whisper
import torch
import soundfile as sf
import librosa
import pyrubberband as pyrb
from num2words import num2words
import re
from pathlib import Path

#-----------------------------------------------------------
# Utilites
#-----------------------------------------------------------

# Function for deleting files in a folder using a template
# Example:
# delete_files_from_dir('/path/to/your/folder', 'chunk*.wav')
def delete_files_from_dir(directory: str, pattern: str) -> None:
    dir_path = Path(directory)

    if not dir_path.is_dir():
        print(f"❌ Error: The path '{directory}' is not an existing directory.")
        return

    # We search only for files matching the chunk*.wav pattern
    for file_path in dir_path.glob(pattern):
        if file_path.is_file():
            try:
                file_path.unlink()  # Delete file
                print(f"✅ Deleted: {file_path}")
            except PermissionError:
                print(f"❌ No rights to delete: {file_path}")
            except Exception as e:
                print(f"❌ Error while deleting {file_path}: {e}")


DECADE_RE = re.compile(r"\b(\d{3,4})('?s)\b", re.IGNORECASE)
ORDINAL_RE = re.compile(r"\b(\d+)(st|nd|rd|th)\b", re.IGNORECASE)
NUMBER_RE = re.compile(r"\b\d+\b")

def convert_decade(match, lang='en'):
    year = int(match.group(1))
    # common spoken decade: "nineteen-twenties" for 1920s
    s = str(year).zfill(4)
    first = int(s[:2])           # e.g., 19
    second = int(s[2:])          # e.g., 20
    # convert first part (19 -> "nineteen")
    first_words = num2words(first, lang=lang)
    # convert decade label (20 -> "twenties"); handle 00 as "hundreds" rare case
    if second == 0:
        # e.g., 1900s -> "nineteen-hundreds"
        second_words = "hundreds"
    elif second == 10:
        second_words = "teens"
    else:
        # produce base like "twenty" -> "twenties" (simple heuristic)
        base = num2words(second, lang=lang)
        if base.endswith("y"):
            second_words = base[:-1] + "ies"
        else:
            second_words = base + "s"
    return f"{first_words}-{second_words}"


def convert_ordinal(match, lang='en'):
    n = int(match.group(1))
    return num2words(n, to='ordinal', lang=lang)


def convert_number(match, lang='en'):
    n = int(match.group(0))
    return num2words(n, lang=lang)


# Example
# s = "In the 1920s people said 1920th and the year 1999 was wild. He was 21 years old."
# print(convert_text(s))
def convert_text(text, lang='en'):
    # 1) decades like 1920s
    text = DECADE_RE.sub(lambda m: convert_decade(m, lang=lang), text)
    # 2) ordinal suffixes like 1920th
    text = ORDINAL_RE.sub(lambda m: convert_ordinal(m, lang=lang), text)
    # 3) remaining bare numbers
    text = NUMBER_RE.sub(lambda m: convert_number(m, lang=lang), text)
    return text


def format_srt_time(seconds):
    """Converts seconds to SRT time format HH:MM:SS,ms."""
    millis = int(seconds * 1000)
    ss, ms = divmod(millis, 1000)
    mm, ss = divmod(ss, 60)
    hh, mm = divmod(mm, 60)
    return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"


def unload_ollama_model(model_name):
    """
    Sends a request to the Ollama API to unload a specific model.
    """
    ollama_endpoint = "http://localhost:11434/api/generate"
    payload = {"model": model_name, "keep_alive": 0}
    try:
        response = requests.post(ollama_endpoint, json=payload)
        response.raise_for_status()
        print(f"✅ Successfully unloaded model: {model_name}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"❌ ERROR: Failed to unload model {model_name}. Is Ollama running? Details: {e}")
        return False


def ollama_translate(text, to_language="Russian", prompt_translate="", is_batch=False):
    """
    Translates a given text to "to_language" using a local Ollama model.
    If is_batch is True, it handles a batch of texts separated by '[---]'.
    """
    ollama_endpoint = "http://localhost:11434/api/generate"

    if is_batch:
        #prompt = f"Translate the following text to {to_language}, keeping the '[-------]' separator between each translated segment. Translate digits and numbers as words. Provide only the translation. Here is the text: '{text}'"
        prompt = f"Translate the following text to {to_language}, preserving the number of lines. Don't split long lines into several lines. Never combine multiple lines into one. Translate digits and numbers as words. Provide only the translation. {prompt_translate} Here is the text: '{text}'"
    else:
        prompt = f"Translate the following text to {to_language}, providing only the translation. {prompt_translate} Here is the text: '{text}'"

    payload = {"model": "translategemma:12b", "stream": False, "prompt": prompt}
    try:
        # Increased timeout for potentially long batch translations
        response = requests.post(ollama_endpoint, json=payload, timeout=180)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        response_json = response.json()
        translated_text = response_json.get("response", "").strip()
        # Sometimes the model might include the original text or quotes, let's try to remove them
        if translated_text.startswith('"') and translated_text.endswith('"'):
            translated_text = translated_text[1:-1]
        return translated_text
    except requests.exceptions.RequestException as e:
        return f"❌ ERROR: Translation failed. Is Ollama running? Details: {e}"


#-------------------------------------------------------------------
# Main
#-------------------------------------------------------------------
def main(
    audio_file,
    whisper_model_name,
    language,
    to_language,
    output_dir,
    perform_translation,
    run_tts,
    normalize_input,
    bypass_input,
    tts_output_dir,
    track_number,
    prompt_translate,
    prompt_transcribe,
    temperature
):
    audio_file_abs = os.path.abspath(audio_file)
    audio_file_origin = audio_file

    # Create output directories first.
    os.makedirs(output_dir, exist_ok=True)
    delete_files_from_dir(output_dir, 'chunk*.wav')
    if run_tts:
        os.makedirs(tts_output_dir, exist_ok=True)
        delete_files_from_dir(tts_output_dir, 'tts_chunk*.wav')

    # Use the main 'output_dir' for the temporary extracted audios.
    # Filename for transcribing:
    temp_audio_filename = f"{os.path.splitext(os.path.basename(audio_file))[0]}_extracted.wav"
    extracted_audio_path = os.path.join(output_dir, temp_audio_filename)
    # Filename for origin audio (to mix after):
    temp_audio_filename_origin = f"{os.path.splitext(os.path.basename(audio_file))[0]}_extracted_origin.wav"
    extracted_audio_path_origin = os.path.join(output_dir, temp_audio_filename_origin)
    # Filename for TTS audio reference:
    temp_audio_filename_reference = f"{os.path.splitext(os.path.basename(audio_file))[0]}_extracted_reference.wav"
    extracted_audio_path_reference = os.path.join(output_dir, temp_audio_filename_reference)

    # Filename for final result audio file:
    final_audio_filename = f"{os.path.splitext(os.path.basename(audio_file))[0]}_final.wav"
    final_audio_path = os.path.join(output_dir, final_audio_filename)
    # Filename for final result audio file mixed with original background:
    final_with_bg_audio_filename = f"{os.path.splitext(os.path.basename(audio_file))[0]}_final_bg.wav"
    final_with_bg_audio_path = os.path.join(output_dir, final_with_bg_audio_filename)

    # If the flag 'bypass_input' is not specified, then extract the audio track and convert it in three versions:
    if not bypass_input:
        command = [
            "ffmpeg",
            "-i", audio_file_abs,
            "-map", f"0:a:{track_number}",
            "-ar", "16000",
            #"-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
            *(["-af", "afftdn=nf=-30:nt=w,highpass=f=100,loudnorm=I=-16:TP=-1.5:LRA=11"] if normalize_input else []), # Normalize & Noise reduction
            #*(["-af", "afftdn=nf=-30:nt=w,highpass=f=100"] if normalize_input else []), # Noise reduction
            #"-af", "pan=mono|c0=c2,loudnorm=I=-16:TP=-1.5:LRA=11",
            #"-af", "pan=mono|c0=c2,afftdn=nf=-30:nt=w,highpass=f=100,loudnorm=I=-16:TP=-1.5:LRA=11",
            "-ac", "1",
            "-acodec", "pcm_s16le",
            "-y",
            "-loglevel", "error",
            extracted_audio_path
        ]

        command_origin = [
            "ffmpeg", "-i", audio_file_abs,
            "-map", f"0:a:{track_number}",
            "-ar", "48000",
            "-ac", "2",
            "-acodec", "pcm_s16le",
            "-y",
            "-loglevel", "error",
            extracted_audio_path_origin
        ]

        command_reference = [
            "ffmpeg", "-i", audio_file_abs,
            "-map", f"0:a:{track_number}",
            "-ar", "24000",
            "-ac", "1",
            "-acodec", "pcm_s16le",
            "-y",
            "-loglevel", "error",
            extracted_audio_path_reference
        ]

        try:
            subprocess.run(command, check=True)
            subprocess.run(command_origin, check=True)
            subprocess.run(command_reference, check=True)
            print(f"\n✅ Successfully extracted and convertion audio to: {extracted_audio_path}\n")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"❌ ERROR: Failed to extract audio. ffmpeg error: {e}")
            print("Please ensure ffmpeg is installed and the video file is valid.")
            return
    # If the flag 'bypass_input' is passed, then use the audio track as is. !!! It doesn't work with video file !!!
    else:
        print("📄 Bypassing convertion and normalization of input audio!")
        extracted_audio_path = audio_file_abs
        extracted_audio_path_origin = audio_file_abs
        extracted_audio_path_reference = audio_file_abs

    audio_file = extracted_audio_path
    audio_file_origin = extracted_audio_path_origin

    # Whisper transcribing load model
    print(f"🔄 Loading Whisper model '{whisper_model_name}'...")
    whisper_model = whisper.load_model(whisper_model_name, device="cuda")

    # Transcribing
    print(f"📄 Transcribing {audio_file} with Whisper...")
    #
    # initial_prompt: Prompt for whispering in the original language. For example, for English: "Use full punctuation, avoid breaks in short phrases. Logical integrity is maintained until a natural pause. ". End with a full stop and a space. Default: None.
    #
    if temperature:
        whisper_result = whisper.transcribe(whisper_model, audio_file, language=language, task="transcribe", word_timestamps=True, hallucination_silence_threshold=0.5, initial_prompt=prompt_transcribe, temperature=0.8)
    else:
        whisper_result = whisper.transcribe(whisper_model, audio_file, language=language, task="transcribe", word_timestamps=True, hallucination_silence_threshold=0.5, initial_prompt=prompt_transcribe)

    # Filtering out the string with special symbols, that is non-translatable
    whisper_segments_raw = whisper_result["segments"]
    pattern = re.compile(r'^[ .¶¦🎵]*$') # '+' -> '*' To exclude empty segments
    w_s_pre1 = [s for s in whisper_segments_raw if not pattern.match(s['text'])]
    # Filtering out the dublicates of string
    whisper_segments = [item for i, item in enumerate(w_s_pre1) if i == 0 or item['text'] != w_s_pre1[i - 1]['text']]

    # Print segments with timestamps
    print(f"📄 Transcribed segments:")
    for segment in whisper_segments:
        print(f"[{segment['start']:.2f}s -> {segment['end']:.2f}s] {segment['text']}")

    # Unload whisper model & free cuda memory
    del whisper_model
    torch.cuda.empty_cache()
    print("🔄 Unloaded Whisper model.")

    # If 'perform_translation' flag is set - to translate into {args.to_language} via ollama with model 'translategemma:12b'
    if perform_translation:
        print("\n📄 Translating all segments at once...")
        #separator = "[-------]"
        separator = "\n"
        all_texts = [segment["text"].strip() for segment in whisper_segments]
        combined_text = convert_text(separator.join(all_texts)) #separator.join(all_texts) # convert_text(): numbers, years -> words
        print(f"\n📄 Text to translate:\n\n{combined_text}\n")

        # Perform translation
        # While number of translated segments match original number of segments
        max_translation_attempts = 30
        for i in range(max_translation_attempts):
            translated_blob = ollama_translate(combined_text, to_language, prompt_translate, is_batch=True)
            print(f"\n📄 Translated:\n\n{translated_blob}\n")

            # Put translated pieces of text into whisper segments
            if translated_blob.startswith("ERROR:"):
                print(f"❌ Warning: Batch translation failed. {translated_blob}")
                for segment in whisper_segments:
                    segment["text_translated"] = "Translation failed."
                sys.exit(1) # Stop program
            else:
                translated_texts = translated_blob.split(separator)
                # Remove last empty element
                translated_texts = translated_texts[:-1] if translated_texts and not translated_texts[-1] else translated_texts
                if len(translated_texts) == len(whisper_segments):
                    for i, segment in enumerate(whisper_segments):
                        segment["text_translated"] = translated_texts[i].strip()
                    break # Break attempts, all is OK
                else:
                    print(
                        f"⚠️ Warning: Mismatch between translated segments ({len(translated_texts)}) and original segments ({len(whisper_segments)}). Retrying..."
                    )
                    if i == max_translation_attempts - 1: # If last iteration - then fallback
                        for i, segment in enumerate(whisper_segments):
                            if i < len(translated_texts):
                                segment["text_translated"] = translated_texts[i].strip()
                            else:
                                segment["text_translated"] = "Translation missing."
                        print("❌ The number of segments in the original and translation does not match. Stop program.")
                        unload_ollama_model("translategemma:12b")
                        sys.exit(1)

        unload_ollama_model("translategemma:12b")
        print(f"✅ The translation was completed successfully.")

    # Audio chunking for audio reference and prepare tasks for TTS
    print("\n📄 Audio Chunking and preparing tasks for batch TTS...\n")
    tts_tasks = [] # to place into json-file for tts-subroutine
    final_mix_inputs = [] # array of TTS-ed result files with start times
    segment_counter = 1
    for segment in whisper_segments:
        start = segment["start"]
        end = segment["end"] + 0.1 # Whisper sometimes mistakes in an end defining as a rule
        text = segment["text"]

        # FFmpeg is used for chunking.
        # This is the chunk filename for the whisper segment.
        chunk_filename = os.path.abspath(os.path.join(output_dir, f"chunk_{segment_counter}.wav"))
        command = [ "ffmpeg", "-i", extracted_audio_path_reference, "-ss", str(start), "-to", str(end), "-c", "copy", "-y", "-loglevel", "error", chunk_filename, ]
        subprocess.run(command, check=True)

        print(f"[{segment_counter}] Whisper:({start:.2f}–{end:.2f}): {text}")

        if perform_translation:
            translated_text = segment.get(
                "text_translated", "Translation not available."
            )
            print(f"    └── RUS: {translated_text.ljust(60)}")

            if run_tts:
                ref_audio_for_tts = chunk_filename
                ref_text_for_tts = text

                print(f"    Using ref_audio: {ref_audio_for_tts}, ref_text: '{ref_text_for_tts}'")

                if (
                    translated_text != "Translation not available."
                    and translated_text != "Translation failed."
                    and translated_text != "Translation missing."
                ):
                    tts_output_file = os.path.abspath(os.path.join(tts_output_dir, f"tts_chunk_{segment_counter}.wav"))
                    tts_tasks.append({
                        "text": translated_text,
                        "output_file": tts_output_file,
                        "source_duration": end - start,
                        "ref_audio": ref_audio_for_tts,
                        "ref_text": ref_text_for_tts,
                    })
                    final_mix_inputs.append((tts_output_file, start))

        segment_counter += 1

    # External script 'phrase-tts.py' is used for dubbing the all of segments.
    if run_tts and tts_tasks:
        # Saving 'tts_tasks.json' file for 'phrase-tts.py'
        tasks_data = {
            "tts_tasks": tts_tasks
        }
        tasks_file = os.path.abspath(os.path.join(tts_output_dir, "tts_tasks.json"))
        with open(tasks_file, 'w', encoding='utf-8') as f:
            json.dump(tasks_data, f, indent=4)

        # Running a script 'phrase-tts' for all segments at once
        tts_command = [
            "python",
            "phrase-tts.py",
            "--tasks_file",
            tasks_file,
            "--language",
            to_language,
        ]
        print(f"\n📄 Starting batch TTS generation...\n")
        subprocess.run(tts_command, check=True)

        if final_mix_inputs:
            print("\n📄 Combining all TTS chunks into a single file (with overlap correction)...\n")

            tts_durations = {}
            for filepath, _ in final_mix_inputs:
                if os.path.exists(filepath):
                    try:
                        info = sf.info(filepath)
                        tts_durations[filepath] = info.duration
                    except Exception as e:
                        print(f"⚠️ Warning: Could not get duration for {filepath} using soundfile. Error: {e}")
                        tts_durations[filepath] = 0
                else:
                    # This case is handled below, just initialize.
                    tts_durations[filepath] = 0

            adjusted_mix_inputs = []
            previous_adjusted_end_time = 0.0
            previous_filepath = ""
            for filepath, original_start_time in final_mix_inputs:
                tts_duration = tts_durations.get(filepath, 0)
                if tts_duration == 0:
                    if os.path.exists(filepath):
                        print(f"⚠️ Warning: TTS chunk {filepath} has zero duration, skipping.")
                    else:
                        print(f"⚠️ Warning: TTS chunk {filepath} not found, skipping it in final mix.")
                    continue

                adjusted_start_time = original_start_time

                if previous_adjusted_end_time > original_start_time:
                    # Calculation of the rate of possible compression of tts-ed audio
                    max_overlap_time = 1.0 # Seconds
                    max_compression_rate = 1.3
                    previous_original_duration = sf.info(previous_filepath).duration
                    desired_compression_rate = 1.0 + (previous_adjusted_end_time - original_start_time) / previous_original_duration
                    compression_rate = min(desired_compression_rate, max_compression_rate)
                    # Compressing previous audio tts_chunk
                    #y, sr = librosa.load(previous_filepath, sr=None)
                    y, sr = sf.read(previous_filepath)
                    #y_changed = librosa.effects.time_stretch(y, rate=max_compression_rate)
                    y_changed = pyrb.time_stretch(y, sr, compression_rate)
                    sf.write(previous_filepath, y_changed, sr)
                    print(f"✅ Saved {previous_filepath} (Speed: {compression_rate}x)")
                    previous_new_duration = sf.info(previous_filepath).duration
                    previous_adjusted_end_time -= (previous_original_duration - previous_new_duration)

                    if previous_adjusted_end_time > original_start_time:
                        overlap = previous_adjusted_end_time - original_start_time
                        increase = min(overlap, max_overlap_time)
                        print(f"📄 Overlap of {overlap:.2f}s detected for {os.path.basename(filepath)}. Increasing start time by {increase:.2f}s to reduce overlap.")
                        adjusted_start_time = original_start_time + increase

                adjusted_mix_inputs.append((filepath, adjusted_start_time))
                previous_adjusted_end_time = adjusted_start_time + tts_duration
                previous_filepath = filepath

            ffmpeg_inputs = []
            filter_chains = []
            mix_inputs_str = ""
            valid_inputs = 0

            for i, (filepath, start_time) in enumerate(adjusted_mix_inputs):
                ffmpeg_inputs.extend(["-i", filepath])
                delay_ms = int(start_time * 1000)
                filter_chains.append(f"[{valid_inputs}]adelay={delay_ms}|{delay_ms}[a{valid_inputs}]")
                mix_inputs_str += f"[a{valid_inputs}]"
                valid_inputs += 1

            if valid_inputs > 0:
                amix_filter = f"{mix_inputs_str}amix=inputs={valid_inputs}:normalize=0[aout]"
                filter_complex = ";".join(filter_chains) + ";" + amix_filter

                output_filename = final_audio_path

                final_command = [
                    "ffmpeg",
                    *ffmpeg_inputs,
                    "-filter_complex",
                    filter_complex,
                    "-map",
                    "[aout]",
                    "-y",
                    "-loglevel", "error",
                    output_filename
                ]

                print(f"\n📄 Creating final combined audio: {output_filename}...")
                try:
                    subprocess.run(final_command, check=True)
                    print(f"✅ Successfully created final audio file {output_filename}.")

                    # Mix with original audio at half volume
                    mixed_output_filename = final_with_bg_audio_path
                    mix_command = [
                        "ffmpeg",
                        "-i", os.path.abspath(audio_file_origin), #os.path.abspath(audio_file),
                        "-i", output_filename,
                        "-filter_complex", "[0:a]volume=0.3[a0]; [1:a]volume=1.0[a1]; [a0][a1]amix=inputs=2:duration=longest:normalize=true[out]",
                        "-map", "[out]",
                        "-y",
                        "-loglevel", "error",
                        mixed_output_filename
                    ]
                    print(f"📄 Creating final mixed audio with background: {mixed_output_filename}...")
                    try:
                        subprocess.run(mix_command, check=True)
                        print(f"✅ Successfully created final mixed audio {mixed_output_filename}.")
                    except subprocess.CalledProcessError as e_mix:
                        print(f"❌ ERROR: Failed to mix audios. ffmpeg error: {e_mix}")

                except subprocess.CalledProcessError as e:
                    print(f"❌ ERROR: Failed to create final combined audio file. ffmpeg error: {e}")
            else:
                print("⚠️ No valid TTS chunks to mix. Skipping final audio creation.")

    if perform_translation:
        print(f"📄 Generating {to_language} SRT subtitles...")
        srt_content = []
        segment_index = 1
        for segment in whisper_segments:
            translated_text = segment.get("text_translated", "").strip()

            if not translated_text or translated_text in ["Translation failed.", "Translation missing."]:
                continue

            start_time = format_srt_time(segment["start"])
            end_time = format_srt_time(segment["end"])

            srt_content.append(str(segment_index))
            srt_content.append(f"{start_time} --> {end_time}")
            srt_content.append(translated_text)
            srt_content.append("")

            segment_index += 1

        if srt_content:
            # Final SRT path
            srt_filename = f"{os.path.splitext(os.path.basename(audio_file_abs))[0]}.srt"
            srt_path = os.path.join(output_dir, srt_filename)

            try:
                with open(srt_path, 'w', encoding='utf-8') as f:
                    f.write("\n".join(srt_content))
                print(f"✅ Successfully created SRT file: {srt_path}")
            except IOError as e:
                print(f"❌ ERROR: Failed to write SRT file. Error: {e}")
        else:
            print("⚠️ No translated text found to generate SRT file.")

    print("-------------------------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Transcribing, translating and dubbing (TTS) of a short audio/video file (Recommendation: containing no more than 20-30 phrases for correct translation and dubbing)."
    )

    parser.add_argument(
        "--audio_file",
        type=str,
        required=True,
        help="Path to the audio/video file to process.",
    )
    parser.add_argument(
        "--whisper_model",
        type=str,
        default="large-v3",
        help="Name of the Whisper model to use. Default = 'large-v3'",
    )
    parser.add_argument(
        "--language",
        type=str,
        #default="English",
        help="Language of --audio_file for Whisper transcribing (en/zh/ru/etc...). Default = 'auto'",
    )
    parser.add_argument(
        "--to_language",
        type=str,
        default="Russian",
        help="Language to translate and TTS into. Default = 'Russian'. The next cross-languages are supported by TTS engine: Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, and Italian.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="chunks",
        help="Directory to save the temporary audio chunks. Default = 'chunks'",
    )
    parser.add_argument(
        "--translate",
        action="store_true",
        help="Is translation into --to_language required? Default = False. The local version of Ollama (model: translategemma:12b) will be used. Create '<output_dir>/<audio_fiel>.srt' file with translated subtitles.",
    )
    parser.add_argument(
        "--run_tts",
        action="store_true",
        help="Enable text-to-speech for translated text. Default = False. The script 'phrase-tts.py' will be used. Use this flag in conjunction with --translate flag.",
    )
    parser.add_argument(
        "--normalize_input",
        action="store_true",
        help="Enable volume normalization of input audio. Default = False.",
    )
    parser.add_argument(
        "--bypass_input",
        action="store_true",
        help="Disable extracting from video and audio conversion. Auidio will be used as is (without normalization too). Don't use this flag with video file (only audio is acceptable)!",
    )
    parser.add_argument(
        "--tts_output_dir",
        type=str,
        default="tts_chunks",
        help="Directory to save the temporary generated TTS audio files. Defaul = 'tts_chunks'.",
    )
    parser.add_argument(
        "--track_number",
        type=str,
        default="0",
        help="Audio track number in video file. Default = 0",
    )
    parser.add_argument(
        "--prompt_translate",
        type=str,
        default="",
        help="Additional prompt for translator (in English) with trailing full stop."
    )
    parser.add_argument(
        "--prompt_transcribe",
        type=str,
        help="Prompt for whispering in the original language. For example, for English: 'Use full punctuation, avoid breaks in short phrases. Logical integrity is maintained until a natural pause. '. End with a full stop and a space. Default: None."
    )
    parser.add_argument(
        "--temperature",
        action="store_true",
        help="Run whisper transcribe with increased temperature parameter (0.8)."
    )

    args = parser.parse_args()

    main(
        args.audio_file,
        args.whisper_model,
        args.language,
        args.to_language,
        args.output_dir,
        args.translate,
        args.run_tts,
        args.normalize_input,
        args.bypass_input,
        args.tts_output_dir,
        args.track_number,
        args.prompt_translate,
        args.prompt_transcribe,
        args.temperature
    )
