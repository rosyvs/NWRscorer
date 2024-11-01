#!/bin/bash

cd ../Audio
for f in *.mp3; do ffmpeg -i "${f}" -vn -c:a pcm_s16le  -ar 16000 -ac 1 "${f%.*}.wav" ; done 