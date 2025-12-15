from pydub import AudioSegment
import os

# Path to your input WAV file
input_file = "source_pena.wav"

# Output directory
output_dir = "slices"
os.makedirs(output_dir, exist_ok=True)

# Load the audio file
audio = AudioSegment.from_wav(input_file)

# Duration of each slice in milliseconds (30 seconds)
slice_duration = 30 * 1000  

# Total number of slices
num_slices = len(audio) // slice_duration + (1 if len(audio) % slice_duration != 0 else 0)

for i in range(num_slices):
    start = i * slice_duration
    end = start + slice_duration
    slice_audio = audio[start:end]
    output_file = os.path.join(output_dir, f"slice_{i+1}.wav")
    slice_audio.export(output_file, format="wav")
    print(f"Saved {output_file}")

print("All slices saved successfully.")
