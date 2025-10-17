@echo off
REM ============================================================
REM Fresh Conda Environment Setup for Qwen-7B Fine-tuning
REM For RTX 4060 8GB with CUDA 12.1
REM ============================================================

echo.
echo ============================================================
echo  Fresh Conda Environment Setup for Qwen Fine-tuning
echo  Target GPU: RTX 4060 8GB with CUDA 12.1
echo ============================================================
echo.

REM Step 1: Remove old environment
echo [1/6] Removing old conda environment...
call conda deactivate 2>nul
call conda env remove -n qwen_finetune -y
echo  - Old environment removed
echo.

REM Step 2: Clean conda cache
echo [2/6] Cleaning conda cache...
call conda clean --all -y
echo  - Conda cache cleaned
echo.

REM Step 3: Clean pip cache
echo [3/6] Cleaning pip cache...
pip cache purge
echo  - Pip cache cleaned
echo.

REM Step 4: Create fresh environment with Python 3.10
echo [4/6] Creating fresh conda environment with Python 3.10...
call conda create -n qwen_finetune python=3.10 -y
if errorlevel 1 (
    echo ERROR: Failed to create conda environment
    pause
    exit /b 1
)
echo  - Environment created successfully
echo.

REM Step 5: Activate environment
echo [5/6] Activating environment...
call conda activate qwen_finetune
if errorlevel 1 (
    echo ERROR: Failed to activate environment
    pause
    exit /b 1
)
echo  - Environment activated
echo.

REM Step 6: Install packages
echo [6/6] Installing required packages...
echo.
echo This will take 5-10 minutes...
echo.

REM Install PyTorch with CUDA 12.1 support
echo  - Installing PyTorch with CUDA 12.1...
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 (
    echo ERROR: Failed to install PyTorch
    pause
    exit /b 1
)

REM Install core dependencies
echo  - Installing transformers and accelerate...
pip install transformers==4.36.2
pip install accelerate==0.25.0

echo  - Installing PEFT for LoRA...
pip install peft==0.7.1

echo  - Installing bitsandbytes for quantization...
pip install bitsandbytes==0.41.3

echo  - Installing datasets...
pip install datasets==2.16.1

echo  - Installing other utilities...
pip install scipy sentencepiece protobuf

echo.
echo ============================================================
echo  Installation Complete!
echo ============================================================
echo.

REM Verify installation
echo Verifying installation...
python -c "import torch; print(f'PyTorch Version: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

echo.
echo ============================================================
echo  Setup completed successfully!
echo  
echo  To activate the environment, run:
echo  conda activate qwen_finetune
echo  
echo  Then you can run the training script:
echo  python scripts/finetune_qwen7b.py
echo ============================================================
echo.
pause