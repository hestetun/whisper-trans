import os
import datetime
import whisper
from whisper.utils import get_writer

def main():
    filepath = input("What file do you want to transcribe?\n")
    output_dir = os.path.dirname(filepath)
    print("Please wait...")

    model = whisper.load_model("large-v2")
    result = model.transcribe(filepath, verbose=True, fp16=False, language="Norwegian", patience=2, beam_size=5)

    # Generate output file name
    base_filename = os.path.splitext(os.path.basename(filepath))[0]
    timestamp = datetime.datetime.now().strftime("%y%m%d")
    output_path = os.path.join(f"{base_filename}_{timestamp}")

    # Write output to SRT file
    srt_writer = get_writer("srt", output_dir)
    srt_writer(result, output_path)
    print(f"Transcription done using OpenAI Whisper. Output file saved to {output_dir}.")

main()
