import os
import whisper
from whisper.utils import get_writer

filename = input("Enter the name of the video file to transcribe: ")
input_directory = input("Enter the directory where the video file is located (default is current directory): ") or "."
input_file = f"{input_directory}/{filename}"

model_name = "large" # or whatever model you prefer
model = whisper.load_model(model_name)
result = model.transcribe(input_file)

translation_model = whisper.load_model("translation")
translation_result = translation_model.translate(transcription_result, target_language="en")

# Save as an SRT file with the model name in the filename
output_directory = input_directory
output_filename = os.path.splitext(filename)[0] + "_" + model_name + "_en.srt"
output_file = os.path.join(output_directory, output_filename)
srt_writer = get_writer("srt", output_directory)
srt_writer(result, output_file)
