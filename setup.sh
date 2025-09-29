#!/bin/bash

set -e

# 모델 파일 다운로드
git clone https://huggingface.co/geonmin-kim/llm_vlm_onnx_sample
cd llm_vlm_onnx_sample
git lfs pull
cd ..