@echo off
pyinstaller --clean --onefile --windowed --add-data lca.ico:. --icon lca.ico --splash lca.png  -d bootloader lca.py
