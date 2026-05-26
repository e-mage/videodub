# Project VideoDub - Video Dubbing

## Description

The **VideoDub** project is designed for automatically dubbing audio tracks into another language.

The project's main idea is to use the [Qwen3-TTS/Base](https://github.com/QwenLM/Qwen3-TTS#voice-clone) model to voice phrases in another language while preserving the speaker's voice timbre and intonation. The model supports the following languages: Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, and Italian.

The project also uses the following locally installed libraries/models/utilities:

- Whisper (transcription),
- SBERT (scenes segmentation),
- Ollama/TranslateGemma (translating scene text into another language),
- FFMpeg (conversion, splicing, and audio extraction).

The dubbing process in this project consists of the following steps:

- The main project launch script, `video-dub.py`, performs the initial detection of the film's semantic scenes based on the provided subtitles (or the result of automatic audio-to-subtitle transcription using the `openai-whisper` library). The semantic scene detector uses the **SBERT embeddings** library and models, which allows searching for sentences with similar meanings. The detector does not use video footage, so an audio track can be passed instead of a video file (also via the `--video` argument to the `video-dub.py` script).

- Scene transcription, scene translation into another language, and dubbing of each phrase in the scene. If the `--no-cut` flag is not specified when running the `video-dub.py` script, the script cuts each detected audio scene and sends it to another script, `scene-dub.py`, for transcription, translation, and dubbing (TTS). The `video-dub.py` script compiles the results of all scene dubbing into one final audio track.

- For batch dubbing all of the phrases in the scene (Text-to-Speech), the `scene-dub.py` script uses the auxiliary script `phrase-tts.py`.

In short, the **VideoDub** project consists of three Python scripts:

- `video-dub.py` (Main script / Scene detector)
- `scene-dub.py` (Transcribes and translates a short scene. Optionally dubs with 'phrase-tts.py')
- `phrase-tts.py` (TTS dubbing of phrases list)

## Installation

### Clone project repo

```bash
git clone https://github.com/e-mage/videodub.git
cd videodub
```

### Create virtual environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### Install Qwen3-TTS

Read: <https://github.com/QwenLM/Qwen3-TTS>

To install `qwen3-tts`:

```bash
pip install -U qwen-tts
MAX_JOBS=4 pip install -U flash-attn --no-build-isolation
```

>**Notice!!!** Installing the `flash-attn` library takes a long time, and also takes up a lot of RAM and CPU. Therefore, limit your computer's resource consumption by set environment variable `MAX_JOBS=4`.

#### Test Qwen3-TTS

Run a web service:

```bash
qwen-tts-demo Qwen/Qwen3-TTS-12Hz-1.7B-Base --ip 0.0.0.0 --port 8000
```

And then open the link `http://<your-ip>:8000` in your browser.

### Install OpenAI-Whisper

```bash
pip install openai-whisper
```

### Install sentence_transformers (aka SBERT)

The SBERT library used to find similarity between sentences. Read:

<https://www.sbert.net/>

<https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2>

To install:

```bash
pip install -U sentence_transformers
```

### Install other python dependencies

```bash
pip install -U librosa soundfile requests pyrubberband num2words pysrt numpy scikit-learn sentencesplit regex
```

### Install ffmpeg

Your operating system must have the ffmpeg package installed. To install:

```bash
sudo apt install ffmpeg
```

### Install ollama and download the translategemma:12b model

<https://ollama.com/>

<https://ollama.com/library/translategemma>

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull translategemma:12b
```

## Required resources, dubbing speed

The project was tested on a computer with 16 GB of RAM and an NVidia card with 16 GB of VRAM. With these parameters, the dubbing time is typically half the length of the audio track being processed.

Without graphics card support, the dubbing process will take significantly longer.

## Usage

### TL;DR

To run the full dubbing process:

```bash
python video-dub.py --video my-chinese-video.mp4 --output out-russian --to-language Russian --max-scene-duration 60
```

To create only subtitles in the original language (`transcribed.srt`) and a list of semantic scenes (`scenes_list.txt`), use the `--no-cut` flag:

```bash
python video-dub.py --video my-chinese-video.mp4 --output out-russian --to-language Russian --max-scene-duration 60 --no-cut
```

> **Please note!** You can interrupt the script at any time in the terminal (Ctrl-C). Re-running the `video-dub.py` script will continue from where it was interrupted.

Sometimes one or more scenes may not be processed correctly, which the script will inform you of at the end. Re-running the script often resolves this issue.

### Results

The following files will be created in the folder specified by the `--output` parameter:

- `transcribed.srt` - Full transcribed subtitles in the original language (SRT format). If the `--translate` parameter was passed during startup, the subtitles will be in English. When running the `video-dub.py` script again, the subtitle file will be reused, meaning re-transcription will not be performed.
- `scenes_list.txt` - A list of scenes with start and end times, as well as the first phrase in each scene. When running the `video-dub.py` script again, the file containing the list of scenes will be reused, meaning re-scanning will not be performed.
- `source_audio.wav` - The extracted stereo audio track from the video file. Can also be reused on subsequent runs.
- `central_channel.wav` - Extracted central channel for cases where the video file contains a multichannel audio track (AC3 - 5.1, 7.1). The central channel typically contains all the dialogue in films, so it is the only channel used for dubbing.
- `merged_audio.wav` - Mono audio track containing the dubbing, i.e., all phrases voiced using TTS, merged together.
- `merged_subtitles.srt` - File with translated subtitles in SRT format.
- `final_mixed_audio.wav` - Final stereo track containing the full dubbing. This is the original audio track (`source_audio.wav` or `central_channel.wav`) and the dubbed track (`merged_audio.wav`) mixed together. The volume of the original audio track is reduced by -12 dB.

> There may be other temporary files from the last scene left in the results folder.

At the moment, the `video-dub.py` script does NOT automatically add the created dubbing track and translated subtitles to the original video file. But this can be done using the `ffmpeg` utility, for example, for dubbing from Chinese to Russian:

```bash
cd ./out-russian
ffmpeg -i final_mixed_audio.wav final_mixed_audio.aac
ffmpeg -i ../my-chinese-video.mp4 -i final_mixed_audio.aac -i transcribed.srt -i merged_subtitles.srt -map 0:v -map 0:a -map 1:a -map 2:s -map 3:s -c copy -metadata:s:a:0 language=zho -metadata:s:a:1 language=rus -metadata:s:s:0 language=zho -metadata:s:s:1 language=rus my-chinese-video-rusdub.mkv
```

### Description of all parameters for the `video-dub.py` script

```bash
python video-dub.py --help

usage: video-dub.py [-h] [--srt SRT] --video VIDEO [--audio-track AUDIO_TRACK] [--output OUTPUT] [--threshold THRESHOLD] [--model MODEL] [--whisper-model WHISPER_MODEL] [--prompt-transcribe PROMPT_TRANSCRIBE] [--language LANGUAGE] [--to-language TO_LANGUAGE] [--translate] [--no-cut] [--max-scene-duration MAX_SCENE_DURATION] [--prompt-translate PROMPT_TRANSLATE]

Video/Audio scene detector based on existing subtitles (SRT-format) (or transcribed from audio with Whisper) using the SBERT embedding model. It also starts the process of dubbing into another language.

options:
-h, --help            show this help message and exit
--srt SRT             Path to the subtitle file (.srt).
                      If not specified, attempts to use the existing file '<--output>/transcribed.srt'.
                      Otherwise, transcribes from audio to the file 'transcribed.srt' and uses it.
--video VIDEO         Path to a video file (the audio track from the video file will be used) or audio file.
--audio-track AUDIO_TRACK
                      Audio track number in the video, default=0 (0 = first).
--output OUTPUT       Output folder. Default = 'scenes_output'.
                      Found scenes are saved in the file 'scenes_list.txt'.
                      Transcribed subtitles are saved in the 'transcribed.srt' file.
--threshold THRESHOLD
                      Sensitivity threshold (0.3-0.9). Higher = fewer scenes.
--model MODEL         SBERT embeddings model (default = 'paraphrase-multilingual-MiniLM-L12-v2', multilingual).
--whisper-model WHISPER_MODEL
                      Whisper model, default = 'large-v3'
                      (Options: tiny/base/small/medium/large/large-v2/large-v3).
--prompt-transcribe PROMPT_TRANSCRIBE
                      Prompt for whispering in the original language.
                      For example, for English: "Use full punctuation, avoid breaks in short phrases. Logical integrity is maintained until a natural pause."
                      End with a period and a space. Default: None.
--language LANGUAGE   Audio language for Whisper (ru/en/etc.). Default = 'auto'.
--to-language TO_LANGUAGE
                      Language to dub into. Default = 'Russian'
                      (Options: Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian).
--translate           Translate audio into English immediately during transcription (option for whisper: translate). Only makes sense with the --no-cut option.
--no-cut              Only parse and create a 'scenes_list.txt' file with scene timestamps,
                      without cutting audio and subsequent dubbing.
--max-scene-duration MAX_SCENE_DURATION
                      Maximum duration of one scene in seconds.
                      Long scenes will be split approximately accordingly. Default = no limit.
--prompt-translate PROMPT_TRANSLATE
                      Additional prompt for the translator (prompt in English, with a trailing full stop).
```

## FAQ

> Why is the video broken down into semantic scenes? Why can't it be transcribed, translated, and voiced in its entirety?

- The main limitation here is translating the text into another language. Translation models like translategemma typically have a very small context window that can't accommodate the entire video/podcast text. Translating individual phrases results in a complete loss of context for the translator.

> Is it even possible to clone timbre and intonation into another language?

- Yes, it is. The developers of the `qwen3-tts` model themselves state this. However, it doesn't always work perfectly. Sometimes, prosodic elements from the original language creep into the cloned version, making it sound as if the person speaks the foreign language well but can't shake their native accent.

> Why does the speaker's timbre sometimes "float" from phrase to phrase, as if different people were speaking?

- Unfortunately, it's impossible to lock the timbre when cloning different phrases in the `qwen3-tts/Base` model. The model is good, but not yet perfect.

> Why are some phrases sometimes missing from the original subtitles or translated subtitles or dubbing audio?

- A missing phrase in the subtitles is most likely due to quiet, unintelligible, or noisy speech in the original audio. However, if a phrase is present in the subtitles but not in the dubbed version, it means the TTS model failed to handle the voiceover. See the next question.

> If I don't like the dubbing in a particular scene, is it possible to redubbed just that scene?

- Yes, it is. To do this, delete all files associated with the specific scene number NNN (`scene_NNN*.*`) in the results folder and run the script again. **Please note** that the dubbing process will work slightly differently each time — including transcription, translation, and voiceover (TTS). To find out the scene number, look in the `scenes_list.txt` file.
