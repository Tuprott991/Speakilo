"""Export and validate a quantized SenseVoiceSmall ONNX runtime."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import onnx
import onnxruntime as ort
from onnx import TensorProto, helper


REQUIRED_FILES = (
    "model_quant.onnx",
    "config.yaml",
    "am.mvn",
    "chn_jpn_yue_eng_ko_spectok.bpe.model",
    "tokens.json",
    "configuration.json",
)


def patch_torch_212_less_type(model_path: Path) -> bool:
    """Repair the float/int Range comparison emitted by Torch 2.12."""
    model = onnx.load(model_path)
    target = next(
        (
            node
            for node in model.graph.node
            if node.op_type == "Less"
            and node.name == "node_lt"
            and list(node.input) == ["arange", "convert_element_type_default"]
        ),
        None,
    )
    if target is None:
        return False

    cast_output = "arange_int64_onevoice"
    if any(cast_output in node.output for node in model.graph.node):
        return False

    cast = helper.make_node(
        "Cast",
        ["arange"],
        [cast_output],
        name="onevoice_cast_arange_int64",
        to=TensorProto.INT64,
    )
    target_index = list(model.graph.node).index(target)
    model.graph.node.insert(target_index, cast)
    target.input[0] = cast_output
    onnx.checker.check_model(model)
    onnx.save(model, model_path)
    return True


def validate_model(model_path: Path, threads: int) -> None:
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )


def export(source: Path, output: Path, threads: int) -> None:
    from funasr_onnx import SenseVoiceSmall

    try:
        SenseVoiceSmall(
            str(source),
            batch_size=1,
            quantize=True,
            intra_op_num_threads=threads,
        )
    except Exception:
        generated = source / "model_quant.onnx"
        if not generated.is_file():
            raise
        if not patch_torch_212_less_type(generated):
            raise
        validate_model(generated, threads)

    output.mkdir(parents=True, exist_ok=True)
    for filename in REQUIRED_FILES:
        source_file = source / filename
        if not source_file.is_file():
            raise FileNotFoundError(f"Missing exported SenseVoice artifact: {source_file}")
        shutil.copy2(source_file, output / filename)
    validate_model(output / "model_quant.onnx", threads)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        help="PyTorch SenseVoiceSmall directory. Downloads iic/SenseVoiceSmall when omitted.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/sensevoice-small-onnx"),
    )
    parser.add_argument("--threads", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source
    if source is None:
        from modelscope import snapshot_download

        source = Path(snapshot_download("iic/SenseVoiceSmall"))
    export(source.resolve(), args.output.resolve(), max(1, args.threads))
    print(f"SenseVoice INT8 ONNX ready at {args.output.resolve()}")


if __name__ == "__main__":
    main()
