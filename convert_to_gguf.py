"""
tinyLLM 폴더에 저장된 Hugging Face 형식 모델
    -> llama.cpp 에서 실행할 수 있는 GGUF 파일로 변환

GGUF 는 로컬 LLM 실행 프로그램인 llama.cpp 계열에서 많이 쓰는 모델 파일 형식
"""
# convert_to_gguf.py
# Robust HuggingFace → GGUF conversion using subprocess
# Run: python convert_to_gguf.py

import subprocess
from pathlib import Path
import sys

# -------- CONFIG --------
LLAMA_CPP_DIR = Path(r"llama.cpp")      # CHANGE THIS
HF_MODEL_DIR = Path(r"tinyLLM")         # CHANGE THIS
OUTPUT_GGUF = Path(r"tinyLLM.gguf")     # CHANGE THIS
OUTTYPE = "f16"
# ------------------------

CONVERTER = LLAMA_CPP_DIR / "convert_hf_to_gguf.py"

if not CONVERTER.exists():
    raise FileNotFoundError(f"convert_hf_to_gguf.py not found at {CONVERTER}")

cmd = [
    sys.executable,
    str(CONVERTER),
    str(HF_MODEL_DIR),
    "--outfile",
    str(OUTPUT_GGUF),
    "--outtype",
    OUTTYPE,
]

print("Running GGUF conversion:")
print(" ".join(cmd))
print()

subprocess.check_call(cmd)

print("\n✅ GGUF conversion complete")
print("Output:", OUTPUT_GGUF)