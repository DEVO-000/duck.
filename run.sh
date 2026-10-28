#!/usr/bin/env bash
# DocScan — обработка фото документов
# Примеры:
#   ./run.sh photo.jpg
#   ./run.sh photo.jpg -o result --debug --ocr
#   ./run.sh папка_с_фото -o result --mode strong
cd "$(dirname "$0")"
python3 -m docscan.cli "$@"
