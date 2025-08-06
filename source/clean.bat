@echo off
echo Cleaning project...

REM Delete .exe, .obj, .exp, and .lib files
del /q *.exe *.obj *.exp *.lib *.ilk *.pdb >nul 2>&1

echo Done.