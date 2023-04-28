import whisper
from whisper.utils import get_writer

def main():
    model = whisper.load_model("medium", device="cpu")
    result = model.transcribe("/Users/ole/Downloads/tst1.wav", task = "translate", verbose = "True", fp16 = False, language = "ar")

    srt_writer = get_writer("srt", "/Volumes/temp/whisper_srt")
    srt_writer(result, "tst1")



    print(result["text"])

main()



#with open("/Volumes/temp/whisper_srt/tst1.srt", "w", encoding="utf-8") as srt_file:
#        write_srt(result["segments"], file=srt_file)