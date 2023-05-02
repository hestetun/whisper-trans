import whisper
from whisper.utils import get_writer

def main():
    filepath = input("What file do you want to transcribe?\n")
    lang = input("What language do you want to transcribe to English? Input using ISO 639-1 codes. e.g Norwegian or no.\n")
    print("Please wait...")

    model = whisper.load_model("large-v2")
    result = model.transcribe(filepath, task = "translate", verbose = "True", fp16 = False, language = lang, temperature=0)

    srt_writer = get_writer("srt", "/Volumes/temp/whisper_srt")
    srt_writer(result, "tst1")

    print(f"Transcription done using OpenAI Whisper from {lang} to English. \nTime spent: ")

main()