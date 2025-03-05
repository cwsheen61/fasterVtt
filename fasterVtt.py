import os
import sys
import subprocess
import re
import time
from datetime import datetime
import openai
from faster_whisper import WhisperModel
from tqdm import tqdm  # Import progress bar library
import shlex
from google.cloud import translate_v2 as translate

# ─────────────────────────────────────────────
# ✅ Fix OpenMP DLL Conflicts
# ─────────────────────────────────────────────
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.add_dll_directory("C:\\Users\\cwshe\\anaconda3\\Library\\bin")
os.environ["PATH"] = ";".join(
    [p for p in os.environ["PATH"].split(";") if "CloudComPy310" not in p]
)

# ─────────────────────────────────────────────
# ✅ Fetch OpenAI API Key
# ─────────────────────────────────────────────
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    print("❌ ERROR: OpenAI API key is missing.")
    exit(1)

# ✅ Fetch available OpenAI models
try:
    models = openai.models.list()
    available_models = [model.id for model in models.data]
except openai.OpenAIError as e:
    print("❌ ERROR: Failed to fetch OpenAI models.", e)
    exit(1)

# ✅ Pick the best available model
preferred_models = ["gpt-4o", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"]
selected_model = next((m for m in preferred_models if m in available_models), "gpt-3.5-turbo")
print(f"✅ Using OpenAI model: {selected_model}")

# ─────────────────────────────────────────────
# ✅ Helper Functions
# ─────────────────────────────────────────────
def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def format_timestamp(seconds):
    """Convert seconds to VTT format hh:mm:ss.mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02}:{minutes:02}:{secs:06.3f}".replace(".", ",")


def get_audio_duration(file_path):
    """Retrieve the duration of an audio file using FFmpeg."""
    result = subprocess.run(
        f'c:\\ffmpeg\\bin\\ffprobe -i "{file_path}" -show_entries format=duration -v quiet -of csv="p=0"',
        shell=True, capture_output=True, text=True
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        return None

def run_ffmpeg_with_progress(command, input_file=None, cwd=None):
    """Runs FFmpeg with progress tracking and estimated time to completion."""
    print(f"{now()} ⏳ Running FFmpeg: {' '.join(command)}")
    total_duration = get_audio_duration(input_file) if input_file else None
    
    process = subprocess.Popen(
        command + ["-progress", "pipe:1", "-v", "error"],
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=cwd  # ✅ Set working directory
    )
    
    progress_bar = None
    if total_duration:
        progress_bar = tqdm(total=total_duration, unit="s", desc="🔄 FFmpeg Processing", dynamic_ncols=True)
    
    for line in process.stdout:
        match = re.search(r'out_time_ms=(\d+)', line)
        if match:
            elapsed_time = int(match.group(1)) / 1_000_000  # Convert microseconds to seconds
            if progress_bar:
                progress_bar.update(elapsed_time - progress_bar.n)
    
    if progress_bar:
        progress_bar.close()
    
    process.wait()  # Ensure it fully completes before returning
    return process.returncode


# ─────────────────────────────────────────────
# ✅ File Path Setup
# ─────────────────────────────────────────────
video_input_file = sys.argv[1]
path, base_filename = os.path.split(video_input_file)
base_name = os.path.splitext(base_filename)[0]

audio_file = os.path.join(path, f"{base_name}.wav")
whisper_output_vtt = os.path.join(path, f"{base_name}.vtt")
cleaned_vtt_file = os.path.join(path, f"{base_name}_fixed.vtt")
prepped_vtt_file = os.path.join(path, f"{base_name}_prepped.vtt")
translated_vtt_file = os.path.join(path, f"{base_name}_translated.vtt")
validated_vtt_file = os.path.join(path, f"{base_name}_validated.vtt")
srt_output_file = os.path.join(path, f"{base_name}.srt")
video_output_file = os.path.join(path, f"{base_name}_subtitled.mp4")
small_video_output_file = os.path.join(path, f"{base_name}_wechat.mp4")

# ─────────────────────────────────────────────
# ✅ Step 1: Extract Audio from Video
# ─────────────────────────────────────────────
if not os.path.exists(audio_file):
    print(f"{now()} 🎙️ Extracting audio from video using FFmpeg...")

    # FFmpeg command must be passed as a LIST (not a string)
    ffmpeg_command = [
        "c:\\ffmpeg\\bin\\ffmpeg", "-y", "-v", "error",
        "-i", video_input_file,
        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
        audio_file
    ]

    # ✅ Run FFmpeg with tqdm progress bar
    total_duration = get_audio_duration(video_input_file)
    process = subprocess.Popen(
        ffmpeg_command + ["-progress", "pipe:1", "-v", "error"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    progress_bar = None
    if total_duration:
        progress_bar = tqdm(total=total_duration, unit="s", desc="🔄 Extracting Audio", dynamic_ncols=True)

    for line in process.stdout:
        match = re.search(r'out_time_ms=(\d+)', line)
        if match:
            elapsed_time = int(match.group(1)) / 1_000_000  # Convert microseconds to seconds
            if progress_bar:
                progress_bar.update(elapsed_time - progress_bar.n)

    if progress_bar:
        progress_bar.close()

    process.wait()  # Ensure it fully completes before returning

    if os.path.exists(audio_file):
        print(f"{now()} ✅ Successfully extracted audio: {audio_file}")
    else:
        print(f"{now()} ❌ FFmpeg audio extraction failed.")

# ─────────────────────────────────────────────
# ✅ Step 2: Transcribe Audio to VTT
# ─────────────────────────────────────────────

if not os.path.exists(whisper_output_vtt):
    print(f"{now()} 📝 Transcribing audio to VTT format...")
    model = WhisperModel("large", device="cuda", compute_type="float16")

    print(f"{now()} ⏳ Starting transcription process...")
    segments, _ = model.transcribe(audio_file)
    segments = list(segments)  # Convert generator to a list

    print(f"{now()} ✅ Transcription process completed. Found {len(segments)} segments.")

    with open(whisper_output_vtt, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")

        # ✅ Initialize tqdm progress bar
        with tqdm(total=len(segments), unit="segments", desc="🎙️ Writing VTT", dynamic_ncols=True) as pbar:
            for segment in segments:
                start_time = format_timestamp(segment.start)
                end_time = format_timestamp(segment.end)
                f.write(f"{start_time} --> {end_time}\n{segment.text.strip()}\n\n")
                pbar.update(1)  # Update progress bar

    print(f"{now()} ✅ Transcription saved: {whisper_output_vtt}")


# ─────────────────────────────────────────────
# ✅ Step 3: Strict Grammar Correction in VTT + Timestamp Fixing
# ─────────────────────────────────────────────

cleaned_vtt_file = os.path.join(path, f"{base_name}_fixed.vtt")

if not os.path.exists(cleaned_vtt_file):
    print(f"{now()} ✨ Fixing grammar and timestamps in {whisper_output_vtt}...")

    with open(whisper_output_vtt, "r", encoding="utf-8") as infile:
        transcript = infile.readlines()  # Read line by line to maintain structure

    fixed_lines = []

    def refine_sentence(sentence):
        """Refine a single sentence for grammar without altering meaning."""
        if not sentence.strip():  # Ignore empty lines
            return sentence
        
        response = openai.chat.completions.create(
            model=selected_model,
            messages=[
                {"role": "system", "content": "You are a strict grammar editor. Do not change meaning, do not add context, do not elaborate. Only correct grammar, punctuation, and minor fluency issues. Return only the corrected sentence, nothing else."},
                {"role": "user", "content": sentence}
            ],
            temperature=0.1  # Reduce creativity to avoid paraphrasing
        )
        return response.choices[0].message.content.strip()

    def fix_timestamp_format(line):
        """Ensure timestamp format is correct (e.g., '00:00:12.500 --> 00:00:15.000')."""
        pattern = r"(\d{1,2}):(\d{1,2}):(\d{1,2})[.,](\d{1,3}) --> (\d{1,2}):(\d{1,2}):(\d{1,2})[.,](\d{1,3})"
        match = re.match(pattern, line)
        if match:
            return f"{match.group(1):02}:{match.group(2):02}:{match.group(3):02}.{match.group(4):03} --> {match.group(5):02}:{match.group(6):02}:{match.group(7):02}.{match.group(8):03}"
        return line  # Return unmodified if not a timestamp

    with tqdm(total=len(transcript), unit="line", desc="🔄 Processing", dynamic_ncols=True) as pbar:
        for line in transcript:
            line = line.strip()  # Remove leading/trailing whitespace

            if line.startswith("WEBVTT"):  # Keep header as is
                fixed_lines.append(line)

            elif "-->" in line:  # Timestamp line
                fixed_lines.append(fix_timestamp_format(line))

            elif line:  # Subtitle line, apply strict grammar correction
                fixed_lines.append(refine_sentence(line))

            else:  # Preserve blank lines
                fixed_lines.append("")

            pbar.update(1)

    with open(cleaned_vtt_file, "w", encoding="utf-8") as outfile:
        outfile.write("\n".join(fixed_lines))  # Maintain line-by-line structure

    print(f"{now()} ✅ Grammar fixed and timestamps corrected. Saved to {cleaned_vtt_file}")

else:
    print(f"{now()} ⏩ Skipping grammar fix, file exists: {cleaned_vtt_file}")


# ─────────────────────────────────────────────
# ✅ Step 4: Google Translate Cleaned VTT
# ─────────────────────────────────────────────
translated_vtt_file = os.path.join(path, f"{base_name}_translated.vtt")

if not os.path.exists(translated_vtt_file):
    print(f"{now()} 🌍 Translating {cleaned_vtt_file} to Chinese...")
    
    with open(cleaned_vtt_file, "r", encoding="utf-8") as infile:
        lines = infile.readlines()
    
    translate_client = translate.Client()
    translated_lines = []
    batch = []
    batch_size = 0
    char_limit = 5000  # Google Translate API character limit per request
    
    with tqdm(total=len(lines), unit="lines", desc="🌍 Translating VTT", dynamic_ncols=True) as pbar:
        for line in lines:
            line = line.strip()
            
            # Keep WEBVTT header and timestamps unchanged
            if line.startswith("WEBVTT") or "-->" in line or line == "":
                if batch:
                    results = translate_client.translate(batch, target_language="zh")
                    translated_lines.extend([res["translatedText"] for res in results])
                    batch = []
                    batch_size = 0
                translated_lines.append(line)
            else:
                batch.append(line)
                batch_size += len(line)
            
            # Send batch to translation API if limit is reached
            if batch_size >= char_limit:
                results = translate_client.translate(batch, target_language="zh")
                translated_lines.extend([res["translatedText"] for res in results])
                batch = []
                batch_size = 0
            
            pbar.update(1)
    
    # Translate any remaining batch
    if batch:
        results = translate_client.translate(batch, target_language="zh")
        translated_lines.extend([res["translatedText"] for res in results])
    
    with open(translated_vtt_file, "w", encoding="utf-8") as outfile:
        outfile.write("\n".join(translated_lines) + "\n")
    
    print(f"{now()} ✅ Translated VTT saved to {translated_vtt_file}")
else:
    print(f"{now()} ⏩ Skipping translation, file exists: {translated_vtt_file}")


# ─────────────────────────────────────────────
# ✅ Step 5: Convert Translated VTT to SRT Using FFmpeg
# ─────────────────────────────────────────────
if not os.path.exists(srt_output_file):
    print(f"{now()} 🎞️ Converting VTT to SRT using FFmpeg...")

    # ✅ Construct FFmpeg command as a list for better subprocess handling
    ffmpeg_command = [
        "c:\\ffmpeg\\bin\\ffmpeg", "-y", "-v", "error",
        "-i", translated_vtt_file,  # Input VTT file
        srt_output_file  # Output SRT file
    ]

    # ✅ Run FFmpeg with subprocess (Avoids shell=True for security)
    result = subprocess.run(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if result.returncode == 0 and os.path.exists(srt_output_file):
        print(f"{now()} ✅ Successfully converted {translated_vtt_file} to {srt_output_file}")
    else:
        print(f"{now()} ❌ FFmpeg VTT to SRT conversion failed. Error:")
        print(result.stderr)  # Show FFmpeg error output
else:
    print(f"{now()} ⏩ Skipping SRT conversion, file exists: {srt_output_file}")

# ─────────────────────────────────────────────
# ✅ Step 6: Burn Subtitles into Video Using FFmpeg (Yellow, Noto Sans CJK)
# ─────────────────────────────────────────────

if not os.path.exists(video_output_file):
    print(f"{now()} 🎞️ Burning styled subtitles into video using FFmpeg...")

    # Get the working directory and subtitle filename
    working_dir = os.path.dirname(srt_output_file)
    srt_filename = os.path.basename(srt_output_file)

    # Construct FFmpeg command as a list (avoids shell escaping issues)
    ffmpeg_command = [
        "c:\\ffmpeg\\bin\\ffmpeg", "-y", "-v", "warning",
        "-i", video_input_file,
        "-vf", f"subtitles={srt_filename}:force_style='FontName=Noto Sans CJK,PrimaryColour=0x00FFFF'",
        "-c:v", "libx264", "-c:a", "copy",
        video_output_file
    ]

    # Get total video duration for progress tracking
    total_duration = get_audio_duration(video_input_file)

    # Start FFmpeg process with progress tracking
    process = subprocess.Popen(
        ffmpeg_command + ["-progress", "pipe:1", "-v", "error"],
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=working_dir  # ✅ Ensures correct file path resolution
    )

    # Initialize tqdm progress bar
    progress_bar = tqdm(total=total_duration, unit="s", desc="🔄 FFmpeg Processing", dynamic_ncols=True)

    for line in process.stdout:
        match = re.search(r'out_time_ms=(\d+)', line)
        if match:
            elapsed_time = int(match.group(1)) / 1_000_000  # Convert microseconds to seconds
            progress_bar.update(elapsed_time - progress_bar.n)

    progress_bar.close()
    process.wait()  # Ensure process completes before checking the output

    # Check if the output video was successfully created
    if os.path.exists(video_output_file):
        print(f"{now()} ✅ Successfully added yellow Noto Sans CJK subtitles to {video_output_file}")
    else:
        print(f"{now()} ❌ FFmpeg subtitle burn-in failed.")
else:
    print(f"{now()} ⏩ Skipping subtitle embedding, file exists: {video_output_file}")

# ─────────────────────────────────────────────
# ✅ Step 7: Optimize Video for WeChat Using FFmpeg (With Progress Bar)
# ─────────────────────────────────────────────
if not os.path.exists(small_video_output_file):
    print(f"{now()} 🎞️ Optimizing video for WeChat using FFmpeg...")

    # ✅ Construct FFmpeg command as a list for better subprocess handling
    ffmpeg_command = [
        "c:\\ffmpeg\\bin\\ffmpeg", "-y", "-v", "error",
        "-i", video_output_file,  # Input video file (with burned-in subtitles)
        "-vf", "scale=960:540",  # Scale to 960x540 for WeChat compatibility
        "-c:v", "libx264", "-crf", "23", "-preset", "fast", "-b:v", "1000k",  # Video encoding
        "-c:a", "aac", "-b:a", "128k",  # Audio encoding
        small_video_output_file  # Output optimized video file
    ]

    # ✅ Get video duration to track progress
    total_duration = get_audio_duration(video_output_file)
    progress_bar = tqdm(total=total_duration, unit="s", desc="🔄 FFmpeg Processing (WeChat Optimization)", dynamic_ncols=True)

    # ✅ Run FFmpeg process with progress tracking
    process = subprocess.Popen(
        ffmpeg_command + ["-progress", "pipe:1"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )

    for line in process.stdout:
        match = re.search(r'out_time_ms=(\d+)', line)
        if match:
            elapsed_time = int(match.group(1)) / 1_000_000  # Convert microseconds to seconds
            progress_bar.update(elapsed_time - progress_bar.n)

    process.wait()
    progress_bar.close()

    # ✅ Check if output file was created
    if process.returncode == 0 and os.path.exists(small_video_output_file):
        print(f"{now()} ✅ Successfully optimized {video_output_file} for WeChat")
    else:
        print(f"{now()} ❌ FFmpeg WeChat optimization failed. Error:")
        print(process.stderr.read())  # Show FFmpeg error output
else:
    print(f"{now()} ⏩ Skipping WeChat optimization, file exists: {small_video_output_file}")
