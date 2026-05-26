#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
video-dub.py

Video/Audio scene detector based on existing subtitles (SRT-format) (or transcribed from audio with Whisper) using the SBERT embedding model. It also starts the process of dubbing into another language.
"""

import os
import sys
import argparse
import subprocess
import tempfile
from datetime import timedelta
import pysrt
import numpy as np
import whisper
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import re
import sentencesplit


# ──────────────────────────────────────────────────────────────
# 1. Subtitles class
# ──────────────────────────────────────────────────────────────

class SubtitleProcessor:
    def __init__(self, srt_path=None):
        self.subs = (
            pysrt.open(srt_path) if srt_path and os.path.exists(srt_path) else None
        )
        self.segments = []

    def _group_segments(self, data_iter, gap_seconds=1.0, min_words=3):
        self.segments = []
        current_group = []
        current_start = None
        current_end = None
        current_text = []

        pattern = re.compile(r'^[ .¶¦🎵]+$') # '+' -> '*' To exclude empty segments
        previous_item_text = ''

        for item in data_iter:
            if hasattr(item, 'text'):
                text = item.text.replace("\n", " ").strip()
                start_sec = item.start.ordinal / 1000.0
                end_sec = item.end.ordinal / 1000.0
            else:
                text = item.get("text", "").replace("\n", " ").strip()
                start_sec = item.get("start", 0)
                end_sec = item.get("end", 0)

            if not text or pattern.match(text) or text == previous_item_text:
                continue

            if current_start is None:
                current_start = start_sec
                current_end = end_sec
                current_text.append(text)
                current_group.append(item)
            else:
                gap = start_sec - current_end
                if gap <= gap_seconds:
                    current_end = end_sec
                    current_text.append(text)
                    current_group.append(item)
                else:
                    if len(" ".join(current_text).split()) >= min_words:
                        self.segments.append(
                            {
                                "start": current_start,
                                "end": current_end,
                                "text": " ".join(current_text),
                                "subs": current_group.copy(),
                            }
                        )
                    current_start = start_sec
                    current_end = end_sec
                    current_text = [text]
                    current_group = [item]

            previous_item_text = text

        if current_text and len(" ".join(current_text).split()) >= min_words:
            self.segments.append(
                {
                    "start": current_start,
                    "end": current_end,
                    "text": " ".join(current_text),
                    "subs": current_group,
                }
            )

        print(f"✅ {len(self.segments)} text segments generated.")
        return self.segments

    def group_subtitles(self, gap_seconds=0.2, min_words=3):
        if not self.subs:
            raise ValueError("The subtitle file is empty or not open.")
        return self._group_segments(self.subs, gap_seconds, min_words)

    def group_from_segments(self, segments, gap_seconds=0.2, min_words=3):
        return self._group_segments(segments, gap_seconds, min_words)


# ──────────────────────────────────────────────────────────────
# 2. Class for audio extraction and transcription
# ──────────────────────────────────────────────────────────────


class AudioProcessor:
    def __init__(self, model_size="large-v3"):
        print(f"🔄 Loading the Whisper model: {model_size} (CUDA)...")
        self.model = whisper.load_model(model_size, device="cuda")
        print("✅ The Whisper model has been loaded.")

    def transcribe(self, audio_path, language=None, translate_to_english=False, initial_prompt=None):
        """Transcribes audio into text with timecodes."""
        print(f"🔄 Transcripting audio (this may take a few minutes)...")

        task = "translate" if translate_to_english else "transcribe"
        result = self.model.transcribe(
            audio_path, language=language, task=task, verbose=False, word_timestamps=True, hallucination_silence_threshold=0.5, initial_prompt=initial_prompt
        )

        segments = []
        for seg in result["segments"]:
            segments.append(
                {"start": seg["start"], "end": seg["end"], "text": seg["text"].strip()}
            )

        print(f"✅ {len(segments)} segments transcribed.")
        return segments

    def segments_to_srt(self, segments, output_path):
        """Saves segments to an SRT file."""
        with open(output_path, "w", encoding="utf-8") as f:
            for i, seg in enumerate(segments, 1):
                start_time = seconds_to_srt_time(seg["start"])
                end_time = seconds_to_srt_time(seg["end"])
                f.write(f"{i}\n{start_time} --> {end_time}\n{seg['text']}\n\n")
        print(f"✅ Subtitles saved: {output_path}")


def seconds_to_srt_time(seconds):
    """Converts seconds to SRT timecode HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


# ──────────────────────────────────────────────────────────────
# 3. Class for scene detection using SBERT
# ──────────────────────────────────────────────────────────────

class SceneDetector:
    def __init__(self, model_name="paraphrase-multilingual-MiniLM-L12-v2"):
        print(f"🔄 Loading the model {model_name}...")
        self.model = SentenceTransformer(model_name)
        print("✅ The model has been loaded.")

    def compute_embeddings(self, segments):
        """Computes vector representations for text segments."""
        texts = [seg["text"] for seg in segments]
        print(f"🔄 Calculating embeddings for {len(texts)} segments...")
        embeddings = self.model.encode(texts, show_progress_bar=True, batch_size=32)
        return embeddings

    def detect_boundaries(self, embeddings, threshold=0.65, window=3):
        """Finds abrupt changes in semantic content"""
        boundaries = []
        for i in range(window, len(embeddings) - window):
            # Comparing average embeddings "before" and "after"
            before = np.mean(embeddings[i-window:i], axis=0)
            after = np.mean(embeddings[i:i+window], axis=0)

            # Cosine distance
            sim = np.dot(before, after) / (np.linalg.norm(before) * np.linalg.norm(after))

            if 1 - sim > threshold:  # A sharp drop in similarity
                boundaries.append(i)

        return boundaries

    def enforce_max_duration(self, segments, boundaries, max_duration):
        """Splits overly long scenes at segment boundaries."""
        stop_tail = 3 # Do not split if the next boundary is less than this value away
        # We add the last segment to the borders, then delete it - this is necessary for the possible breakdown of the last semantic scene
        boundaries.append(len(segments)-1)
        new_boundaries = []
        prev_b = 0
        for b in boundaries:
            if segments[b]["start"] - segments[prev_b]["start"] < max_duration:
                new_boundaries.append(b)
            else:
                prev_s = prev_b
                for s in range(prev_b + 1, b):
                    if segments[s]["start"] - segments[prev_s]["start"] > max_duration:
                        if segments[s-1]["text"][-1] in SENTENCE_END_MARKS and (b - s) >= stop_tail:
                            new_boundaries.append(s)
                            prev_s = s
            prev_b = b
        # Let's remove the previously added border.
        new_boundaries.pop()
        boundaries.pop()
        return new_boundaries

    def create_scenes(self, segments, boundaries):
        """Assembles a final list of scenes based on the boundaries found."""
        scenes = []
        start_idx = 0

        for b_idx in boundaries:
            end_idx = b_idx + 1
            if end_idx > start_idx:
                scenes.append(
                    {
                        "start_time": segments[start_idx]["start"],
                        "end_time": segments[end_idx - 1]["end"],
                        "text_preview": segments[start_idx]["text"][:50] + "...",
                        "segment_count": end_idx - start_idx,
                    }
                )
            start_idx = end_idx

        # Adding the last scene
        if start_idx < len(segments):
            scenes.append(
                {
                    "start_time": segments[start_idx]["start"],
                    "end_time": segments[-1]["end"],
                    "text_preview": segments[start_idx]["text"][:50] + "...",
                    "segment_count": len(segments) - start_idx,
                }
            )

        return scenes


# ──────────────────────────────────────────────────────────────
# 4. Video and Time Utilities
# ──────────────────────────────────────────────────────────────

def seconds_to_timecode(seconds):
    """Converts seconds to a time string HH:MM:SS.ms"""
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def export_edl(scenes, output_path):
    """Exports a list of scenes to a text file (for editing)."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("No.\tBegin\t\tEnd\t\tDuration\tText (preview)\n")
        f.write("-" * 100 + "\n")
        for i, scene in enumerate(scenes):
            start = seconds_to_timecode(scene["start_time"])
            end = seconds_to_timecode(scene["end_time"])
            duration = f"{scene['end_time'] - scene['start_time']:.1f}с"
            text = scene["text_preview"].replace("\t", " ")
            f.write(f"{i + 1}\t{start}\t{end}\t{duration}\t{text}\n")
    print(f"📄 The list of scenes is saved in {output_path}")


def import_scenes_from_file(file_path):
    """Imports scenes from the scenes_list.txt file."""
    scenes = []
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()[2:]  # Skip the title and separator
        for line in lines:
            if not line.strip():
                continue
            parts = line.strip().split("\t")
            if len(parts) >= 4:
                start_time = timecode_to_seconds(parts[1])
                end_time = timecode_to_seconds(parts[2])
                text_preview = parts[4] if len(parts) > 4 else ""
                scenes.append(
                    {
                        "start_time": start_time,
                        "end_time": end_time,
                        "text_preview": text_preview,
                    }
                )
    return scenes


def timecode_to_seconds(timecode):
    """Converts a time string HH:MM:SS.mmm to seconds."""
    parts = timecode.replace(",", ".").split(":")
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    return 0.0


def get_audio_channels(video_path, audio_track=0):
    """Returns the number of channels in an audio track."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        f"a:{audio_track}",
        "-show_entries",
        "stream=channels",
        "-of",
        "csv=p=0",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0 and result.stdout.strip():
        return int(result.stdout.strip())
    return 2


def extract_and_process_full_audio(video_path, output_dir, audio_track=0):
    """Extracts and processes full audio from video. Saves in WAV format.

    Returns:
        Path to the file with extracted audio (center channel or stereo as is)
    """
    channels = get_audio_channels(video_path, audio_track)
    print(f"🔊 The audio track has {channels} channels")

    if channels > 2:  # If the track is multi-channel
        processed_path = os.path.join(output_dir, "center_channel.wav")
    else:  # If it's a stereo track
        processed_path = os.path.join(output_dir, "source_audio.wav")

    if os.path.exists(processed_path):
        print(f"📄 Using existing processed audio: {processed_path}")
        return processed_path

    if (
        channels > 2
    ):  # If it's a multi-channel audio track, extract only the center channel.
        print(
            f"🔄 Multichannel audio ({channels} channels) - extracting the center channel..."
        )
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-map",
            f"0:a:{audio_track}",
            "-af",
            "pan=mono|c0=c2",
            "-acodec",
            "pcm_s16le",
            # "-ar",
            # "48000",
            # "-ac",
            # "1",
            processed_path,
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return processed_path

    else:  # If it's a stereo audio track, we'll take it as is in WAV format.
        print(f"🔄 Stereo audio - extract audio as is from {video_path}...")

        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            # "-vn",
            "-map",
            f"0:a:{audio_track}",
            # "-c:a", #"copy",
            "-acodec",
            "pcm_s16le",
            # "-ar",
            # "48000",
            # "-ac",
            # "1",
            processed_path,
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return processed_path


def cut_audio(
    input_video, # Using as a fallback. In most cases, it uses audio passed through the processed_audio_path parameter.
    scenes,
    output_dir,
    audio_track=0,
    language=None,
    processed_audio_path=None,
    whisper_model="large-v3",
    prompt_translate="",
    prompt_transcribe=None
):
    """Splits audio from video into scenes using ffmpeg."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n✂️  Cutting audio into {len(scenes)} scenes...")

    """For each scene"""
    for i, scene in enumerate(scenes):
        start = scene["start_time"]
        end = scene["end_time"]
        duration = end - start

        if duration < 5:
            continue

        final_wav = os.path.join(output_dir, f"scene_{i + 1:03d}_final.wav")
        if os.path.exists(final_wav):
            print(f"  ⏭️  Scene {i + 1}: already exists ({final_wav}) - skipping")
            continue

        ext = "wav" # Extension of cut file
        output_filename = os.path.join(output_dir, f"scene_{i + 1:03d}.{ext}")

        if processed_audio_path and os.path.exists(processed_audio_path):
            source_input = ["-i", processed_audio_path]
            seek_offset = start
        else:
            source_input = ["-i", input_video, "-map", f"0:a:{audio_track}"]
            seek_offset = start

        cmd = (
            [
                "ffmpeg",
                "-y",
            ]
            + source_input
            + [
                "-ss",
                str(seek_offset),
                "-t",
                str(duration),
                # "-c:a",
                # "pcm_s16le",
                # "-ar",
                # "44100",
                # "-ac",
                # "1",
                output_filename,
            ]
        )

        """We're starting dubbing (transcribing, translate and TTS) for the scene with 'scene-dub.py' script."""
        # First, we try to transcribe with the default temperature; if this fails, then temperature=0.9 is set.
        for attempt in ["regular", "temperature"]:
            try:
                subprocess.run(cmd, check=True, capture_output=True)
                # Run subprocess for Transcribe+Translate+TTS for the scene
                transcribe_tts_command = [
                    "python",
                    "scene-dub.py",
                    "--audio_file",
                    output_filename,
                    "--whisper_model", whisper_model,
                    *(["--language", language] if language else []),
                    "--output_dir",
                    output_dir,
                    "--translate",
                    "--run_tts",
                    "--normalize_input",
                    "--bypass_input", # Use audio to transcribe and TTS without tanscoding
                    "--tts_output_dir",
                    output_dir,
                    "--prompt_translate",
                    prompt_translate,
                    *(["--prompt_transcribe", prompt_transcribe] if prompt_transcribe else []),
                    *(["--temperature"] if attempt=="temperature" else [])

                ]
                print(f"Starting batch Transcribe + Translate + TTS generation... {"!!! Fallback attempt of transcribing with corrected temperature !!!" if attempt=="temperature" else ""}")
                subprocess.run(transcribe_tts_command, check=True)
                print(
                    f"  ✅ Сцена {i + 1}: {seconds_to_timecode(start)} - {seconds_to_timecode(end)} ({duration:.1f}с)"
                )
                break # Successful, skip transcribe attempt with corrected temperature
            except subprocess.CalledProcessError as e:
                print(f"  ❌ Scene cutting error {i + 1}: {e}")
            except FileNotFoundError:
                print("  ❌ FFmpeg not found!")
                sys.exit(1) # Stop program


def merge_audio_scenes(scenes, output_dir, output_filename="merged_audio.wav"):
    """Combines audio files scene_NNN_final.wav into one, taking into account time shifts."""
    audio_files = []
    offsets = []

    for i, scene in enumerate(scenes):
        start = scene["start_time"]
        end = scene["end_time"]
        duration = end - start

        if duration < 5:
            continue

        audio_file = os.path.join(output_dir, f"scene_{i + 1:03d}_final.wav")
        if os.path.exists(audio_file):
            audio_files.append(audio_file)
            offsets.append(start)
            print(f"  📂 Audio scene {i + 1}: {audio_file} (offset: {start:.2f}с)")
        else:
            print(f"  ⚠️  Audio scene {i + 1} not found: {audio_file}")

    if not audio_files:
        print("❌ There are no audio files to combine.")
        return None

    if len(audio_files) == 1:
        import shutil

        output_path = os.path.join(output_dir, output_filename)
        shutil.copy(audio_files[0], output_path)
        print(f"✅ Audio copied (one file): {output_path}")
        return output_path

    output_path = os.path.join(output_dir, output_filename)

    print(f"\n🔄 Merge {len(audio_files)} audio files taking into account offsets...")

    inputs = []
    delay_filters = []

    for i, (audio_file, offset) in enumerate(zip(audio_files, offsets)):
        inputs.extend(["-i", audio_file])
        delay_ms = int(offset * 1000)
        delay_filters.append(f"[{i}:a]adelay={delay_ms}|{delay_ms}[delayed{i}]")

    amix_inputs = "".join([f"[delayed{i}]" for i in range(len(audio_files))])
    filter_complex = (
        ";".join(delay_filters)
        + f";{amix_inputs}amix=inputs={len(audio_files)}:duration=longest:normalize=0,loudnorm=I=-16:TP=-1.5:LRA=11[out]"
    )

    cmd = (
        ["ffmpeg", "-y"]
        + inputs
        + [
            "-filter_complex",
            filter_complex,
            "-map",
            "[out]",
            "-ac",
            "2",
            output_path,
        ]
    )

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ Audio merged: {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"❌ Audio merging error: {e}")
        if e.stderr:
            print(f"   stderr: {e.stderr}")
        return None


def merge_subtitle_scenes(scenes, output_dir, output_filename="merged_subtitles.srt"):
    """Combines SRT files scene_NNN.srt into one, taking into account time shifts."""
    srt_files = []
    offsets = []

    for i, scene in enumerate(scenes):
        start = scene["start_time"]
        end = scene["end_time"]
        duration = end - start

        if duration < 5:
            continue

        srt_file = os.path.join(output_dir, f"scene_{i + 1:03d}.srt")
        if os.path.exists(srt_file):
            srt_files.append(srt_file)
            offsets.append(start)
        else:
            print(f"  ⚠️  SRT scene {i + 1} not found: {srt_file}")

    if not srt_files:
        print("❌ There are no SRT files to merge.")
        return None

    output_path = os.path.join(output_dir, output_filename)

    print(f"\n🔄 Merging {len(srt_files)} SRT files...")

    merged_subs = []

    for srt_file, offset in zip(srt_files, offsets):
        try:
            subs = pysrt.open(srt_file)
            for sub in subs:
                sub.start.ordinal += int(offset * 1000)
                sub.end.ordinal += int(offset * 1000)
            merged_subs.extend(subs)
        except Exception as e:
            print(f"⚠️  Read error: {srt_file}: {e}")

    merged_subs.sort(key=lambda x: x.start.ordinal)

    for i, sub in enumerate(merged_subs, 1):
        sub.index = i

    subs_obj = pysrt.SubRipFile(merged_subs)
    subs_obj.save(output_path, encoding="utf-8")
    print(f"✅ Subtitles merged: {output_path}")
    return output_path


def mix_audio_with_tts(tts_audio_path, output_dir, background_reduction_db=-12):
    """
    Mixes TTS audio with the original audio track (at reduced volume).
    Uses 'source_audio.wav' (stereo) or 'center_channel.wav' (multichannel).
    """
    source_audio = None
    if os.path.exists(os.path.join(output_dir, "source_audio.wav")):
        source_audio = os.path.join(output_dir, "source_audio.wav")
    elif os.path.exists(os.path.join(output_dir, "center_channel.wav")):
        source_audio = os.path.join(output_dir, "center_channel.wav")

    if not source_audio:
        print("⚠️  Source audio track not found. Using TTS only.")
        return tts_audio_path

    if not tts_audio_path or not os.path.exists(tts_audio_path):
        print("⚠️  TTS audio not found.")
        return source_audio

    output_path = os.path.join(output_dir, "final_mixed_audio.wav")

    print(
        f"🔄 Mixing TTS with original audio (reduced by {background_reduction_db} dB)..."
    )

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        source_audio,
        "-i",
        tts_audio_path,
        "-filter_complex",
        f"[0:a]volume={background_reduction_db}dB[bg];[bg][1:a]amix=inputs=2:duration=longest:normalize=0[out]",
        "-map",
        "[out]",
        "-ac",
        "2",
        output_path,
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ Audio mixed: {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"❌ Audio mixing error: {e}")
        if e.stderr:
            print(f"   stderr: {e.stderr}")
        return tts_audio_path


# ──────────────────────────────────────────────────────────────
# 5. Additional utilities
# ──────────────────────────────────────────────────────────────

SENTENCE_END_MARKS = {'.', '!', '?', '。', '？', '！'}

def _ends_with_sentence_ender(text):
    """Checks if the text ends with a sentence end mark."""
    if not text:
        return False
    text = text.strip()
    if not text:
        return False
    segmenter = sentencesplit.Segmenter(language="en")
    sentences = segmenter.segment(text)
    if not sentences:
        return False
    last_sentence = sentences[-1].strip()
    if not last_sentence:
        return False
    return last_sentence[-1] in SENTENCE_END_MARKS



# ──────────────────────────────────────────────────────────────
# 6. Main function
# ──────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Video/Audio scene detector based on existing subtitles (SRT-format) (or transcribed from audio with Whisper) using the SBERT embedding model. It also starts the process of dubbing into another language."
    )
    parser.add_argument(
        "--srt",
        help="Path to the subtitle file (.srt). If not specified, attempts to use the existing file '<--output>/transcribed.srt'.   Otherwise, transcribes from audio to the file 'transcribed.srt' and uses it.",
    )
    parser.add_argument(
        "--video",
        required=True,
        help="Path to a video file (the audio track from the video file will be used) or audio file.",
    )
    parser.add_argument(
        "--audio-track",
        type=int,
        default=0,
        help="Audio track number in the video, default=0 (0 = first).",
    )
    parser.add_argument(
        "--output",
        default="scenes_output",
        help="Output folder. Default = 'scenes_output'. Found scenes are saved in the file 'scenes_list.txt'. Transcribed subtitles are saved in the 'transcribed.srt' file."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="Sensitivity threshold (0.3-0.9). Higher = fewer scenes.",
    )
    parser.add_argument(
        "--model",
        default="paraphrase-multilingual-MiniLM-L12-v2",
        help="SBERT embeddings model (default = 'paraphrase-multilingual-MiniLM-L12-v2', multilingual).",
    )
    parser.add_argument(
        "--whisper-model",
        default="large-v3",
        help="Whisper model, default = 'large-v3' (Options: tiny/base/small/medium/large/large-v2/large-v3).",
    )
    parser.add_argument(
        "--prompt-transcribe",
        type=str,
        help="Prompt for whispering in the original language. For example, for English: 'Use full punctuation, avoid breaks in short phrases. Logical integrity is maintained until a natural pause. '. End with a period and a space. Default: None."
    )
    parser.add_argument(
        "--language",
        help="Audio language for Whisper (ru/en/etc.). Default = 'auto'."
    )
    parser.add_argument(
        "--to-language",
        default="Russian",
        help="Language to dub into. Default = 'Russian' (Available options: Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian)."
    )
    parser.add_argument(
        "--translate",
        action="store_true",
        help="Translate audio into English immediately during transcription (option for whisper: translate). Only makes sense with the --no-cut option.",
    )
    parser.add_argument(
        "--no-cut", action="store_true", help="Only parse and create a 'scenes_list.txt' file with scene timestamps, without cutting audio and subsequent dubbing."
    )
    parser.add_argument(
        "--max-scene-duration",
        type=float,
        default=None,
        help="Maximum duration of one scene in seconds. Long scenes will be split approximately accordingly. Default = no limit.",
    )
    parser.add_argument(
        "--prompt-translate",
        default="",
        help="Additional prompt for the translator (prompt in English, with a trailing full stop)."
    )

    args = parser.parse_args()

    print("🎬 SBERT Scene Detector (Audio Version)")
    print("=" * 50)

    if not os.path.exists(args.video):
        print(f"❌ Video file not found: {args.video}")
        return

    os.makedirs(args.output, exist_ok=True)

    scenes_list_path = os.path.join(args.output, "scenes_list.txt")

    if os.path.exists(scenes_list_path):
        print(f"📄 Scene file found - loading: {scenes_list_path}")
        scenes = import_scenes_from_file(scenes_list_path)
        print(f"✅ Loaded {len(scenes)} scenes from file.")
    else:
        default_srt = os.path.join(args.output, "transcribed.srt")

        if args.srt and os.path.exists(args.srt):
            print(f"📄 Subtitles will be used: {args.srt}")
            processor = SubtitleProcessor(args.srt)
            segments = processor.group_subtitles() # gap_seconds=0.2, min_words=3
        elif os.path.exists(default_srt):
            print(f"📄 Previously saved subtitles will be used: {default_srt}")
            processor = SubtitleProcessor(default_srt)
            segments = processor.group_subtitles() # gap_seconds=0.2, min_words=3
        else:
            print("🎤 No subtitles found - transcribing audio...")
            audio_processor = AudioProcessor(model_size=args.whisper_model)

            # Extract audio track with number=args.audio_track from video
            extracted_audio = extract_and_process_full_audio(args.video, args.output, audio_track=args.audio_track)

            try:
                raw_segments = audio_processor.transcribe(
                    extracted_audio, language=args.language, translate_to_english=args.translate, initial_prompt=args.prompt_transcribe
                )

                audio_processor.segments_to_srt(raw_segments, default_srt) # Save to transcribed.srt

                processor = SubtitleProcessor()
                segments = processor.group_from_segments(raw_segments) # gap_seconds=0.2, min_words=3
            finally:
                del audio_processor
                torch.cuda.empty_cache()
                print("Unloaded Whisper model.")

        if not segments:
            print("❌ Failed to get text segments.")
            return

        detector = SceneDetector(model_name=args.model)
        embeddings = detector.compute_embeddings(segments)
        boundaries = detector.detect_boundaries(embeddings, threshold=args.threshold)

        if args.max_scene_duration:
            print(f"📄 A limit of {args.max_scene_duration} seconds will be used for the duration of one scene.")
            boundaries = detector.enforce_max_duration(segments, boundaries, args.max_scene_duration)

        scenes = detector.create_scenes(segments, boundaries)

        print(f"\n🎯 {len(scenes)} semantic scenes found.")

        durations = [s["end_time"] - s["start_time"] for s in scenes]
        print(f"   Average duration: {np.mean(durations):.1f} sec")
        print(f"   Min: {np.min(durations):.1f} sec, Max: {np.max(durations):.1f} sec")

        export_edl(scenes, scenes_list_path)

    """
    If the --no-cut flag is not specified, then start the process of cutting, transcribing, translating and dubbing (TTS) audio scenes.
    """
    if not args.no_cut:
        if args.video and os.path.exists(args.video):
            processed_audio = extract_and_process_full_audio(
                args.video,
                args.output,
                audio_track=args.audio_track,
            )
            # In the cut_audio() function, a child script is called for transcription, translation and tts
            cut_audio(
                args.video,
                scenes,
                args.output,
                audio_track=args.audio_track,
                language=args.language,
                processed_audio_path=processed_audio,
                whisper_model=args.whisper_model,
                prompt_translate=args.prompt_translate,
                prompt_transcribe=args.prompt_transcribe
            )

            print("\n🔗 Merging audio and subtitles of scenes...")
            merge_audio_scenes(scenes, args.output)
            merge_subtitle_scenes(scenes, args.output)

            tts_audio_path = os.path.join(args.output, "merged_audio.wav")
            final_audio = mix_audio_with_tts(tts_audio_path, args.output)
            print(f"📝 Final audio: {final_audio}")
        else:
            print("⚠️  Video file not found. Skipping cut.")
    else:
        print("🚫 Cutting is disabled with the --no-cut flag.")

    print("\n✅ Ready!")


if __name__ == "__main__":
    main()
