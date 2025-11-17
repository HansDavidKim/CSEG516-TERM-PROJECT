@echo off
REM Quick start script for running RLB-MI attack with Docker on Windows

echo ================================
echo RLB-MI Docker Quick Start
echo ================================
echo.

REM Check if Docker is installed
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Docker is not installed
    echo Please install Docker Desktop: https://docs.docker.com/desktop/install/windows-install/
    pause
    exit /b 1
)

REM Check if docker-compose is installed
docker-compose --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: docker-compose is not installed
    echo Please install docker-compose
    pause
    exit /b 1
)

REM Detect GPU
set USE_GPU=false
nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    echo GPU detected - using GPU version
    set USE_GPU=true
    set PROFILE=gpu
    set SERVICE=rlb-mi-gpu
) else (
    echo No GPU detected - using CPU version
    set PROFILE=cpu
    set SERVICE=rlb-mi-cpu
)

echo.
echo Creating directories...
if not exist "checkpoints" mkdir checkpoints
if not exist "attack_results" mkdir attack_results
if not exist "dataset" mkdir dataset
if not exist "pretrained" mkdir pretrained
echo Directories created

echo.
echo Building Docker image...
docker-compose build %SERVICE%
if %errorlevel% neq 0 (
    echo Error: Failed to build Docker image
    pause
    exit /b 1
)
echo Docker image built successfully

echo.
echo ================================
echo Starting container...
echo ================================
echo.
echo You are now inside the RLB-MI container!
echo To run the attack, use:
echo.
echo   python main.py run-rlb-mi-attack ^
echo     --generator checkpoints/generator_last.pt ^
echo     --target-model checkpoints/vgg16_celeba_best.pt ^
echo     --target-class 0 ^
echo     --episodes 40000
echo.
echo Or run the example script:
echo   python example_rlb_mi.py
echo.

docker-compose --profile %PROFILE% run --rm %SERVICE% bash
