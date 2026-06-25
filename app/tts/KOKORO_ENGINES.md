# OneVoice Kokoro TTS Engines

## Vietnamese output

- Runtime: `kokoro-vietnamese`
- Version in this repo: `0.1.0`
- Source path: `app/tts/Kokoro-Vietnamese/src`
- Default male voice: `hung_thinh`
- Used for: `en2vi`, where the translated output is Vietnamese.

The adapter supports local weights through `tts.kokoro_vi.model_path`,
`voicepack_path`, and `config_path`. If those are empty, the runtime resolves
the default files from `contextboxai/Kokoro-Vietnamese`, which should be
pre-cached before offline edge evaluation.

## English output

- Active runtime: bundled `kokoro` 0.9.4 on PyTorch CUDA
- Optional CPU experiment: Kokoro v1.0 INT8 ONNX
- Source path: `app/tts/Kokoro-Vietnamese/kokoro`
- Default male voice: `am_adam`
- Used for: `vi2en`, where the translated output is English.

Run `scripts/setup_cuda_tts.ps1` in the `onevoice` conda environment. The setup
installs matched CUDA 12.4 PyTorch packages and then restores NumPy 1.26.4
because `funasr-onnx` is not compatible with NumPy 2.x.

The RTX 3050 runtime uses CUDA explicitly. MT and punctuation remain on CPU to
avoid GPU contention. Only one output-language TTS model stays resident, keeping
active TTS allocation near 330 MB on the tested laptop.

## Latency policy

ASR and MT are loaded before the UI is marked ready. TTS warms up in a daemon
thread after that. Committed MT chunks trigger prefetched `/api/speak` requests
only while the UI Voice toggle is enabled. Synthesis is serialized for runtime
safety, while WebSocket ASR and MT remain independent.
