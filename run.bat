@echo off
REM ============================================================
REM  DocScan - обработка фото документов
REM  Примеры:
REM     run.bat photo.jpg
REM     run.bat photo.jpg -o result --debug --ocr
REM     run.bat C:\photos -o result --mode strong
REM ============================================================
cd /d "%~dp0"
python -m docscan.cli %*
pause
