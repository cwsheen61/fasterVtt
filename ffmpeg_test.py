import os
import subprocess

# Simulated Inputs
video_input_file = "c:/Users/cwshe/Zoom-Workspace/PennState_6/ENTR810_02202025.mp4"
srt_output_file = "c:/Users/cwshe/Zoom-Workspace/PennState_6/ENTR810_02202025.srt"
video_output_file = "c:/Users/cwshe/Zoom-Workspace/PennState_6/ENTR810_02202025_sub_test.mp4"

# Extract working directory and file names
working_dir = os.path.dirname(video_input_file)  # "c:/Users/cwshe/Zoom-Workspace/PennState_6"
srt_filename = os.path.basename(srt_output_file)  # "ENTR810_02202025.srt"

# Construct FFmpeg command using relative subtitles path
ffmpeg_command = [
    "c:\\ffmpeg\\bin\\ffmpeg",  # Full path to FFmpeg executable
    "-y",
    "-i", video_input_file,
    "-vf", f"subtitles={srt_filename}:force_style='FontName=Noto Sans CJK,PrimaryColour=0x00FFFF'",
    "-c:v", "libx264",
    "-c:a", "copy",
    video_output_file
]

# Print the command for debugging
print("\n🟡 Constructed FFmpeg Command:\n")
print(" ".join(ffmpeg_command))
print("\n🔄 Now running FFmpeg...\n")

# Run FFmpeg inside the working directory
process = subprocess.run(ffmpeg_command, cwd=working_dir, text=True)

# Check result
if process.returncode == 0 and os.path.exists(video_output_file):
    print("\n✅ FFmpeg successfully burned subtitles into video!")
else:
    print("\n❌ FFmpeg subtitle burn-in failed.")
