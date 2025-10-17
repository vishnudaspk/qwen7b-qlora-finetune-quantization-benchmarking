# Quick Fix for NumPy 2.x and Missing Packages Issue
# Run this in your activated qwen_finetune environment

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " Quick Fix: NumPy 2.x & Missing Packages" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "[1/3] Downgrading NumPy to 1.x..." -ForegroundColor Yellow
pip install "numpy<2.0" --force-reinstall
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to downgrade NumPy" -ForegroundColor Red
    exit 1
}
Write-Host "  ✅ NumPy downgraded successfully" -ForegroundColor Green
Write-Host ""

Write-Host "[2/3] Installing missing tokenizer package (tiktoken)..." -ForegroundColor Yellow
pip install tiktoken
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to install tiktoken" -ForegroundColor Red
    exit 1
}
Write-Host "  ✅ tiktoken installed successfully" -ForegroundColor Green
Write-Host ""

Write-Host "[3/3] Installing additional tokenizer utilities..." -ForegroundColor Yellow
pip install tokenizers sentencepiece
Write-Host "  ✅ Tokenizer utilities installed" -ForegroundColor Green
Write-Host ""

Write-Host "============================================================" -ForegroundColor Green
Write-Host " Fix Complete!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""

Write-Host "Verifying installation..." -ForegroundColor Cyan
python -c "import numpy; print(f'NumPy version: {numpy.__version__}'); import tiktoken; print('tiktoken: OK'); import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"

Write-Host ""
Write-Host "✅ All fixed! You can now run:" -ForegroundColor Green
Write-Host "   python scripts/finetune_qwen7b.py" -ForegroundColor White
Write-Host ""