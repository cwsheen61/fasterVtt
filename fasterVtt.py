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

def get_audio_duration(file_path):
    result = subprocess.run(
        f'c:\\ffmpeg\\bin\\ffprobe -i "{file_path}" -show_entries format=duration -v quiet -of csv="p=0"',
        shell=True, capture_output=True, text=True
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        return None



def run_ffmpeg_with_progress(command, input_file=None):
    """Runs FFmpeg with progress tracking and estimated time to completion."""
    print(f"{now()} ⏳ Running FFmpeg: {command}")

    total_duration = get_audio_duration(input_file) if input_file else None

    # Force FFmpeg to run correctly with `shell=True`
    process = subprocess.Popen(
        command + " -progress pipe:1 -v error",
        shell=True,  # Required for Windows
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
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

def run_ffmpeg(command):
    """Runs an FFmpeg command and shows real-time output."""
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        print(f"{now()} ▶ {line.strip()}")

    process.wait()
    return process.returncode

def clean_up(file):
    if os.path.exists(file):
        print(f"{now()} 🗑️ Deleting {file}...")
        os.remove(file)

# ─────────────────────────────────────────────
# ✅ File Path Setup
# ─────────────────────────────────────────────
video_input_file = sys.argv[1]
path, base_filename = os.path.split(video_input_file)
base_name = os.path.splitext(base_filename)[0]

audio_file = os.path.join(path, f"{base_name}.wav")
whisper_output_vtt = os.path.join(path, f"{base_name}.vtt")
cleaned_vtt_file = os.path.join(path, f"{base_name}_fixed.vtt")
zh_vtt_file = os.path.join(path, f"{base_name}_zh.vtt")
srt_output_file = os.path.join(path, f"{base_name}.srt")
video_output_file = os.path.join(path, f"{base_name}_subtitled.mp4")
small_video_output_file = os.path.join(path, f"{base_name}_wechat.mp4")


# ✅ Step 1: Extract Audio from Video (With Progress)
if not os.path.exists(audio_file):
    print(f"{now()} 🎙️ Extracting audio from video...")
    run_ffmpeg_with_progress(f'"c:\\ffmpeg\\bin\\ffmpeg" -hwaccel cuda -y -nostdin -i "{video_input_file}" -ar 16000 -ac 1 -c:a pcm_s16le "{audio_file}"', video_input_file)

# ─────────────────────────────────────────────
# ✅ Step 2: Run Faster-Whisper for Transcription (Only if Needed)
# ─────────────────────────────────────────────
if not os.path.exists(whisper_output_vtt):
    print(f"{now()} 📝 Running Faster Whisper transcription...")

    model = WhisperModel("large", device="cuda", compute_type="float16")

    # Get total audio duration for progress tracking
    duration_command = f'c:\\ffmpeg\\bin\\ffprobe -i "{audio_file}" -show_entries format=duration -v quiet -of csv="p=0"'
    try:
        total_duration = float(subprocess.run(duration_command, shell=True, capture_output=True, text=True).stdout.strip())
    except ValueError:
        total_duration = None

    # Transcribe and display progress
    segments, _ = model.transcribe(audio_file)

    with open(whisper_output_vtt, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")

        with tqdm(total=total_duration, unit="s", desc="🎙️ Transcribing", dynamic_ncols=True) as pbar:
            for segment in segments:
                f.write(f"{segment.start:.3f} --> {segment.end:.3f}\n{segment.text.strip()}\n\n")
                if total_duration:
                    pbar.update(segment.end - segment.start)  # Update progress bar

    print(f"{now()} ✅ Transcription saved: {whisper_output_vtt}")
else:
    print(f"{now()} ⏩ Skipping transcription, file exists: {whisper_output_vtt}")


# ─────────────────────────────────────────────
# ✅ Step 3: Fix Grammar & Create _fixed.vtt (With Progress Bar)
# ─────────────────────────────────────────────
def fix_grammar_vtt(input_vtt, output_vtt):
    if os.path.exists(output_vtt):
        print(f"{now()} ⏩ Skipping grammar fix: {output_vtt} already exists.")
        return

    print(f"{now()} ✨ Fixing grammar in {input_vtt}...")

    with open(input_vtt, "r", encoding="utf-8") as infile:
        transcript = infile.read()

    sections = transcript.strip().split("\n\n")
    total_sections = len(sections)

    with tqdm(total=total_sections, unit="block", desc="🔄 Grammar Fixing", dynamic_ncols=True) as pbar:
        fixed_sections = []
        
        for section in sections:
            response = openai.chat.completions.create(
                model=selected_model,
                messages=[{"role": "system", "content": "Fix grammar and improve readability while keeping timestamps intact."},
                          {"role": "user", "content": section}],
                temperature=0.3
            )
            fixed_sections.append(response.choices[0].message.content)
            pbar.update(1)

    with open(output_vtt, "w", encoding="utf-8") as outfile:
        outfile.write("\n\n".join(fixed_sections))

    print(f"{now()} ✅ Grammar fixed and saved to {output_vtt}")

fix_grammar_vtt(whisper_output_vtt, cleaned_vtt_file)

# ─────────────────────────────────────────────
# ✅ Step 4: Translate to Chinese (_zh.vtt)
# ─────────────────────────────────────────────
def translate_vtt(input_vtt, output_vtt):
    if os.path.exists(output_vtt):
        print(f"{now()} ⏩ Skipping translation: {output_vtt} already exists.")
        return

    print(f"{now()} 🌎 Translating {input_vtt} to Chinese...")

    with open(input_vtt, "r", encoding="utf-8") as infile:
        transcript = infile.read()

    response = openai.chat.completions.create(
        model=selected_model,
        messages=[{"role": "system", "content": "Translate this subtitle file into Simplified Chinese while keeping timestamps intact."},
                  {"role": "user", "content": transcript}],
        temperature=0.3
    )

    with open(output_vtt, "w", encoding="utf-8") as outfile:
        outfile.write(response.choices[0].message.content)

    print(f"{now()} ✅ Translation saved to {output_vtt}")

translate_vtt(cleaned_vtt_file, zh_vtt_file)

# ✅ Step 5: Convert VTT to SRT
if os.path.exists(zh_vtt_file) and os.path.getsize(zh_vtt_file) > 0:
    print(f"{now()} 🎞️ Converting VTT to SRT using FFmpeg...")
    ffmpeg_cmd = f'"c:\\ffmpeg\\bin\\ffmpeg" -y -i "{zh_vtt_file}" "{srt_output_file}"'
    
    # Debugging Log: Show exactly what command is being run
    print(f"🔹 Running: {ffmpeg_cmd}")

    result = run_ffmpeg_with_progress(ffmpeg_cmd)

    if result == 0:
        print(f"{now()} ✅ Successfully converted {zh_vtt_file} to {srt_output_file}")
    else:
        print(f"{now()} ❌ FFmpeg failed with error code {result}")
else:
    print(f"{now()} ❌ Skipping Step 5: VTT file missing or empty: {zh_vtt_file}")

# 🔄 Step 6: Burn Subtitles into Video with Progress
if not os.path.exists(video_output_file):
    print(f"\n\n{now()}: adding subtitles to {video_input_file}\n")

    # Format path for FFmpeg compatibility
    new_srt_outputFileName = srt_output_file.replace("\\", "\\\\").replace(":", "\\:")

    ffmpeg_cmd = f"""
    c:\\ffmpeg\\bin\\ffmpeg -y -v error -progress pipe:1 -i "{video_input_file}" \
    -vf subtitles='{new_srt_outputFileName}' -c:v libx264 -c:a copy "{video_output_file}"
    """.strip()

    print(f"🔹 Running: {ffmpeg_cmd}")  # Debugging log

    result = run_ffmpeg_with_progress(ffmpeg_cmd, video_input_file)  # 🚀 Uses existing logic

    if result == 0:
        print(f"{now()} ✅ Successfully burned subtitles into {video_output_file}")
    else:
        print(f"{now()} ❌ FFmpeg failed with error code {result}")
else:
    print(f"{now()} ⏩ Skipping subtitle embedding, file exists: {video_output_file}")


# ✅ Step 7: Convert to WeChat Compatible Format
if not os.path.exists(video_small_outputFileName):
    print(f"\n\n{now()}: reducing file size {video_output_file}\n")

    ffmpeg_cmd = f"c:\\ffmpeg\\bin\\ffmpeg -y -i \"{video_output_file}\" -vf scale=960:540 -c:v libx264 -crf 23 -preset fast -b:v 1000k -c:a aac -b:a 128k \"{video_small_outputFileName}\""

    print(f"🔹 Running: {ffmpeg_cmd}")  # Debugging log
    os.system(ffmpeg_cmd)

    if os.path.exists(video_small_outputFileName):
        print(f"{now()} ✅ Successfully optimized video for WeChat: {video_small_outputFileName}")
    else:
        print(f"{now()} ❌ WeChat video optimization failed.")
else:
    print(f"{now()} ⏩ Skipping WeChat optimization, file exists: {video_small_outputFileName}")


print(f"{now()} ✅ All steps complete!")
