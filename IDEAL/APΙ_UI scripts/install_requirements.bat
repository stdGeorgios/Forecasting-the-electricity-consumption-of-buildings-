@echo off
cd /d %~dp0

echo ====================================
echo IDEAL Forecasting Demo Installer
echo ====================================
echo.

echo Checking for Python 3.11...
py -3.11 --version >nul 2>nul
if %errorlevel%==0 goto use_py311

echo Checking for Python 3.12...
py -3.12 --version >nul 2>nul
if %errorlevel%==0 goto use_py312

echo.
echo ERROR: Python 3.11 or 3.12 was not found.
echo Please install Python 3.11 or 3.12 (64-bit) first.
pause
exit /b 1

:use_py311
set PY_CMD=py -3.11
goto continue_install

:use_py312
set PY_CMD=py -3.12
goto continue_install

:continue_install
echo Using %PY_CMD%
%PY_CMD% --version
echo.

if exist venv (
    echo Existing venv found. Removing it...
    rmdir /s /q venv
)

echo Creating virtual environment...
%PY_CMD% -m venv venv
if not exist venv\Scripts\python.exe (
    echo ERROR: Failed to create virtual environment.
    pause
    exit /b 1
)

echo.
echo Upgrading pip...
venv\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 (
    echo ERROR: Failed to upgrade pip.
    pause
    exit /b 1
)

echo.
echo Installing requirements...
venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo ERROR: pip install failed.
    pause
    exit /b 1
)

echo.
echo Running self-test...
venv\Scripts\python.exe -c "import fastapi, uvicorn, pandas, numpy, sklearn, gradio, requests, joblib; print('OK')"
if errorlevel 1 (
    echo ERROR: Self-test failed.
    pause
    exit /b 1
)

echo.
echo ====================================
echo Installation completed successfully.
echo ====================================
pause