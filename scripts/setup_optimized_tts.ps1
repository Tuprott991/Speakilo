$ErrorActionPreference = "Stop"

$modelDir = Join-Path $PSScriptRoot "..\models\kokoro-onnx"
$modelDir = [System.IO.Path]::GetFullPath($modelDir)
New-Item -ItemType Directory -Force $modelDir | Out-Null

python -m pip install "numpy==1.26.4"
python -m pip install --no-deps "kokoro-onnx==0.5.0"
python -m pip install "espeakng-loader>=0.2.4" "phonemizer-fork>=3.3.2"

python -c "from huggingface_hub import hf_hub_download; from pathlib import Path; p=Path(r'$modelDir'); hf_hub_download('speaches-ai/Kokoro-82M-v1.0-ONNX-int8','model.onnx',local_dir=str(p)); hf_hub_download('speaches-ai/Kokoro-82M-v1.0-ONNX-int8','voices.bin',local_dir=str(p))"

python -c "import kokoro_onnx, numpy; print('kokoro-onnx ready; numpy=' + numpy.__version__)"
