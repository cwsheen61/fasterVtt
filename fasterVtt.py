import os
import re
import sys
import subprocess
from datetime import datetime
import shutil
from faster_whisper import WhisperModel

# ─────────────────────────────────────────────
# ✅ Fix OpenMP DLL Conflicts
# ─────────────────────────────────────────────
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Prevent OpenMP conflicts
os.add_dll_directory("C:\\Users\\cwshe\\anaconda3\\Library\\bin")  # Force correct DLL

# ─────────────────────────────────────────────
# ✅ Load OpenAI API Key (If Needed in Future)
# ─────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("❌ ERROR: OpenAI API key not found. Please set the 'OPENAI_API_KEY' environment variable.")

# ─────────────────────────────────────────────
# ✅ Helper Functions
# ─────────────────────────────────────────────
def now():
    """Returns the current timestamp for logging."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def check_ffmpeg():
    """Ensure FFmpeg is installed and accessible."""
    if not shutil.which("c:\\ffmpeg\\bin\\ffmpeg"):
        raise FileNotFoundError("❌ ERROR: FFmpeg not found at expected location.")

def run_command_live_output(command):
    """Runs a shell command and displays real-time output."""
    print(f"{now()} ⏳ Running: {command}")

    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        print(f"{now()} ▶ {line.strip()}")  # Print real-time output

    process.wait()

    if process.returncode == 0:
        print(f"{now()} ✅ Command completed successfully.")
    else:
        print(f"{now()} ❌ Error running command.")

def clean_up(file):
    """Deletes a file if it exists."""
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
small_video_output_file = os.path.join(path, f"{base_name}_small.mp4")

# ─────────────────────────────────────────────
# ✅ Define Command Dictionary (For Debugging)
# ─────────────────────────────────────────────
COMMANDS = {
    "extract_audio": f'c:\\ffmpeg\\bin\\ffmpeg -hwaccel cuda -y -nostdin -i "{video_input_file}" -ar 16000 -ac 1 -c:a pcm_s16le "{audio_file}"',
    "convert_vtt_to_srt": f'c:\\ffmpeg\\bin\\ffmpeg -y -nostdin -i "{zh_vtt_file}" "{srt_output_file}"',
    "embed_subtitles": f'c:\\ffmpeg\\bin\\ffmpeg -hwaccel cuda -y -nostdin -i "{video_input_file}" -vf "subtitles={srt_output_file}" "{video_output_file}"',
    "optimize_video": f'c:\\ffmpeg\\bin\\ffmpeg -hwaccel cuda -y -nostdin -i "{video_output_file}" -vf scale=960:540 -c:v h264_nvenc -crf 23 -preset fast -b:v 1000k -c:a aac -b:a 128k "{small_video_output_file}"'
}

# ─────────────────────────────────────────────
# ✅ Debug: Print Commands Before Execution
# ─────────────────────────────────────────────
print("\n🔹 Debugging COMMANDS dictionary...\n")
for call_name, command_string in COMMANDS.items():
    print(f"{now()} COMMAND: {call_name}\n{command_string}\n")

# ─────────────────────────────────────────────
# ✅ Cleanup Previous Files
# ─────────────────────────────────────────────
clean_up(audio_file)
clean_up(whisper_output_vtt)
clean_up(cleaned_vtt_file)
clean_up(zh_vtt_file)
clean_up(srt_output_file)
clean_up(video_output_file)
clean_up(small_video_output_file)

print(f"{now()} 🔹 Processing Started for Video: {video_input_file}")

check_ffmpeg()  # Ensure FFmpeg is installed

# ─────────────────────────────────────────────
# ✅ Step 1: Extract Audio from Video
# ─────────────────────────────────────────────
print(f"{now()} 🎙️ Extracting audio from video using FFmpeg with CUDA...")
run_command_live_output(COMMANDS["extract_audio"])

# ─────────────────────────────────────────────
# ✅ Step 2: Run Faster-Whisper (Transcription)
# ─────────────────────────────────────────────
print(f"{now()} 📝 Running Faster Whisper for transcription...")

model = WhisperModel("large", device="cuda", compute_type="float16")

segments, _ = model.transcribe(audio_file)

with open(whisper_output_vtt, "w", encoding="utf-8") as f:
    f.write("WEBVTT\n\n")
    for segment in segments:
        f.write(f"{segment.start:.3f} --> {segment.end:.3f}\n{segment.text.strip()}\n\n")

print(f"{now()} ✅ Transcription saved to {whisper_output_vtt}")

# ─────────────────────────────────────────────
# ✅ Step 3: Convert VTT to SRT
# ─────────────────────────────────────────────
print(f"{now()} 🔄 Converting VTT to SRT...")
run_command_live_output(COMMANDS["convert_vtt_to_srt"])

# ─────────────────────────────────────────────
# ✅ Step 4: Embed Subtitles into the Video
# ─────────────────────────────────────────────
print(f"{now()} 🔄 Embedding subtitles into the video...")
run_command_live_output(COMMANDS["embed_subtitles"])

# ─────────────────────────────────────────────
# ✅ Step 5: Optimize Video for WeChat
# ─────────────────────────────────────────────
print(f"{now()} 🔄 Optimizing video size for WeChat...")
run_command_live_output(COMMANDS["optimize_video"])

print(f"{now()} ✅ All steps complete!")
