#!/bin/bash

SOURCE_DIR="samples"
TARGET_DIR="audios"

mkdir -p "$TARGET_DIR"

for file in "$SOURCE_DIR"/*.mp3; 
do
  base_name=$(basename "$file" .mp3)
  
  target_file="$TARGET_DIR/$base_name.mov"
  
  # Convert the .mp3 file to .mov using ffmpeg
  ffmpeg -i "$file" -c:a pcm_s16le -vn "$target_file"
  
  if [ $? -eq 0 ]; then
    echo "Converted: $file -> $target_file"
  else
    echo "Failed to convert: $file"
  fi
done