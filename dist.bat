@echo off
rem pyinstaller --clean --onefile --windowed --add-data lca.ico:. --add-binary lca.ico:. --icon lca.ico --splash lca.png  -d bootloader lca.py
pyinstaller lca.spec
