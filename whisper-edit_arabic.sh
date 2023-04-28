#!/bin/bash

read -p "Enter the directory where the video file is located (default is current directory): " input_file
output_file="${input_file%.*}.srt"

/opt/homebrew/bin/whisper $input_file --language "ar" --task "translate" --fp16 "False" -f "srt" --verbose "True" --model "base" --model_dir "/Volumes/temp/whisper_models" --output_dir "/Volumes/temp/whisper_srt" #--threads 4
