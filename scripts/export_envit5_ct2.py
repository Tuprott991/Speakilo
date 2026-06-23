"""
Export VietAI/envit5-translation to CTranslate2 for faster local inference.

Run from the repo root:
    conda activate onevoice
    python scripts/export_envit5_ct2.py --quantization int8
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Export envit5 to CTranslate2.")
    parser.add_argument("--model", default="VietAI/envit5-translation")
    parser.add_argument("--output-dir", default="models/envit5-ct2-int8")
    parser.add_argument("--quantization", default="int8")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    if os.path.exists(output_dir):
        if not args.force:
            print(f"[CT2 Export] Output already exists: {output_dir}")
            print("[CT2 Export] Use --force to replace it.")
            return 0
        shutil.rmtree(output_dir)

    os.makedirs(output_dir, exist_ok=True)

    try:
        import ctranslate2
        from transformers import AutoTokenizer
    except ImportError as exc:
        print("[CT2 Export] Missing dependency. Install ctranslate2 and transformers first.")
        raise exc

    print(f"[CT2 Export] Source model: {args.model}")
    print(f"[CT2 Export] Output dir  : {output_dir}")
    print(f"[CT2 Export] Quantization: {args.quantization}")

    converter = ctranslate2.converters.TransformersConverter(args.model)
    converter.convert(
        output_dir,
        quantization=args.quantization,
        force=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.save_pretrained(output_dir)
    print("[CT2 Export] Done. Set translation.backend=auto or ctranslate2 in config/config.yaml.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
