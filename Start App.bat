@echo off
setlocal enabledelayedexpansion
title Stock Valuation and Investment Analysis

rem Capture this script's own folder ONCE, here, outside any bracketed block.
rem
rem It must never be written as %~dp0 inside a parenthesised if-block below.
rem cmd.exe expands percent-variables while PARSING a block -- before it
rem decides whether to run it -- so a folder whose name contains a closing
rem bracket ends the block early and aborts the entire script instantly, with
rem the window closing before any message or pause is reached. That is not
rem hypothetical: "stock-valuation-tool-main (1)" is the name Windows gives
rem the folder when this project's .zip is downloaded a second time, and it
rem made the launcher fail with nothing but a black flash.
rem
rem !HERE! is delayed expansion, resolved at execution time, so it is safe
rem inside a block no matter what the folder is called.
set "HERE=%~dp0"

rem "pushd" rather than "cd /d": pushd also works when the project sits on a
rem network drive or UNC path (\\server\share\...), where "cd" silently fails
rem and leaves the script running in C:\Windows\System32.
pushd "%~dp0" 2>nul
if errorlevel 1 (
    echo.
    echo Could not open the folder this file is in:
    echo   !HERE!
    echo.
    echo Try copying the whole project folder to your Desktop and running it from there.
    echo.
    pause
    exit /b 1
)

rem ---------------------------------------------------------------------
rem Check the whole project is here, not just this one file.
rem
rem The most common way people hit this: downloading only "Start App.bat"
rem from GitHub instead of the whole project, or double-clicking it from
rem inside the downloaded .zip without extracting first (Windows will happily
rem run a .bat straight out of a zip, but it copies ONLY that file to a temp
rem folder, so nothing else is there).
rem ---------------------------------------------------------------------
if not exist "requirements.txt" goto missing_files
if not exist "app.py" goto missing_files
if not exist "src" goto missing_files

set PYCMD=
where python >nul 2>nul && set PYCMD=python
if not defined PYCMD (
    where py >nul 2>nul && set PYCMD=py
)
if not defined PYCMD (
    where python3 >nul 2>nul && set PYCMD=python3
)

if not defined PYCMD (
    echo.
    echo Could not find Python on this computer.
    echo.
    echo Install Python 3.10 or newer from https://www.python.org/downloads/
    echo IMPORTANT: on the first screen of the installer, tick
    echo "Add python.exe to PATH" before clicking Install.
    echo.
    echo Then close this window and double-click this file again.
    echo.
    pause
    popd
    exit /b 1
)

rem Python 3.10 is the real minimum: the project uses modern type-hint
rem syntax, and SciPy requires 3.10+. Checked explicitly, because the
rem failure on an older version is a confusing syntax error deep in a
rem library rather than anything that names the actual problem.
%PYCMD% -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
if errorlevel 1 (
    echo.
    echo Your version of Python is too old for this project.
    echo.
    %PYCMD% -c "import sys; print('   You have: Python ' + '.'.join(map(str, sys.version_info[:3])))"
    echo    Needed:   Python 3.10 or newer
    echo.
    echo Install a newer version from https://www.python.org/downloads/
    echo and remember to tick "Add python.exe to PATH" in the installer.
    echo.
    pause
    popd
    exit /b 1
)

echo Using %PYCMD%
%PYCMD% --version
echo.
echo Installing required packages - this can take a minute the first time...
echo.
%PYCMD% -m pip install -r requirements.txt --disable-pip-version-check
if errorlevel 1 (
    echo.
    echo Something went wrong installing the required packages.
    echo.
    echo Things worth trying:
    echo   1. Check you are connected to the internet.
    echo   2. If you are on a work or school computer, its security settings
    echo      may block installing packages - try a personal computer.
    echo   3. Run this again; a failed download will often work second time.
    echo.
    echo If it still fails, the error text above is the useful part - it can be
    echo reported at:
    echo   https://github.com/CadenWiseltier/stock-valuation-tool/issues
    echo.
    pause
    popd
    exit /b 1
)

echo.
echo Starting the app. A browser tab should open automatically at http://localhost:8501
echo Keep this window open while you use the app. Close it (or press Ctrl+C) to stop.
echo.
%PYCMD% -m streamlit run app.py

popd
pause
exit /b 0

rem ---------------------------------------------------------------------
:missing_files
echo.
echo ============================================================
echo   The project files are missing.
echo ============================================================
echo.
echo This launcher is in:
echo   !HERE!
echo.
echo ...but the rest of the project is not there. It needs to sit in the
echo same folder as app.py, requirements.txt and the src folder.
echo.
echo This almost always means one of two things:
echo.
echo   1. Only this one file was downloaded.
echo      On GitHub, use the green "Code" button and choose
echo      "Download ZIP" to get the whole project - not the download
echo      arrow on an individual file.
echo.
echo   2. It was run from inside the .zip without extracting it first.
echo      Right-click the downloaded .zip, choose "Extract All...",
echo      open the extracted folder, and double-click this file there.
echo.
echo Download link:
echo   https://github.com/CadenWiseltier/stock-valuation-tool
echo.
pause
popd
exit /b 1
