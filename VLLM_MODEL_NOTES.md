# vLLM Model Configuration Notes

## Current Setup (ROLLBACK CONFIG)

**Model:** `QuantTrio/Qwen3-VL-30B-A3B-Instruct-AWQ`
**Type:** 30B MoE (Mixture of Experts) with AWQ quantization
**Context:** 32,768 tokens
**Date:** 2026-01-20

### Docker Run Command (for rollback)
```powershell
docker run -d --gpus all `
  --name vllm-multimodal `
  -p 1234:8000 `
  vllm/vllm-openai:latest `
  --model QuantTrio/Qwen3-VL-30B-A3B-Instruct-AWQ `
  --trust-remote-code `
  --max-model-len 32768 `
  --served-model-name qwen3-multimodal `
  --gpu-memory-utilization 0.7 `
  --enable-auto-tool-choice `
  --tool-call-parser hermes
```

---

## New Setup: Qwen3-VL-8B AWQ (smaller model, bigger context)

**Model:** `cyankiwi/Qwen3-VL-8B-Instruct-AWQ-4bit`
**Type:** 8B Dense with 4-bit AWQ quantization
**Context:** Up to 256K native (we'll use 131072 = 128K to start)

### Docker Commands

**1. Stop and remove current container:**
```powershell
docker stop vllm-multimodal
docker rm vllm-multimodal
```

**2. Start new 8B model with extended context:**
```powershell
docker run -d --gpus all `
  --name vllm-multimodal `
  -p 1234:8000 `
  vllm/vllm-openai:latest `
  --model cyankiwi/Qwen3-VL-8B-Instruct-AWQ-4bit `
  --trust-remote-code `
  --max-model-len 65536 `
  --served-model-name qwen3-multimodal `
  --gpu-memory-utilization 0.75 `
  --enable-auto-tool-choice `
  --tool-call-parser hermes
```

### Key Changes
| Setting | Old (30B MoE) | New (8B Dense) |
|---------|---------------|----------------|
| Model | QuantTrio/Qwen3-VL-30B-A3B-Instruct-AWQ | cyankiwi/Qwen3-VL-8B-Instruct-AWQ-4bit |
| Params | 30B (MoE, ~3B active) | 8B (dense) |
| Quant | AWQ | AWQ 4-bit |
| Context | 32,768 | 65,536 (2x more!) |
| GPU Util | 0.7 | 0.75 |

### If you want even more context later:
- `--max-model-len 98304` (96K)
- `--max-model-len 131072` (128K) - may need higher GPU util

### Alternative 8-bit model (slightly larger, possibly better quality):
```powershell
docker run -d --gpus all `
  --name vllm-multimodal `
  -p 1234:8000 `
  vllm/vllm-openai:latest `
  --model cpatonn/Qwen3-VL-8B-Instruct-AWQ-8bit `
  --trust-remote-code `
  --max-model-len 49152 `
  --served-model-name qwen3-multimodal `
  --gpu-memory-utilization 0.75 `
  --enable-auto-tool-choice `
  --tool-call-parser hermes
```
Note: 8-bit uses more VRAM than 4-bit, so context is reduced to 48K here.
