$ErrorActionPreference = "Stop"

$python = Join-Path $env:CONDA_PREFIX "python.exe"
if (-not $env:CONDA_PREFIX -or -not (Test-Path -LiteralPath $python)) {
    throw "Activate the onevoice conda environment before running this script."
}

& $python -m pip install --force-reinstall `
    "torch==2.6.0+cu124" `
    "torchvision==0.21.0+cu124" `
    "torchaudio==2.6.0+cu124" `
    --index-url "https://download.pytorch.org/whl/cu124"

# funasr-onnx requires NumPy 1.x even though torchvision accepts newer releases.
& $python -m pip install --force-reinstall "numpy==1.26.4"

& $python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))"
