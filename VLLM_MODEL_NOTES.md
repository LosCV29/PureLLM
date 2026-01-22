# vLLM Model Configuration Notes

## Current Setup (FP8 - RECOMMENDED)

**Model:** `Qwen/Qwen3-VL-8B-Instruct-FP8`
**Type:** 8B Dense with FP8 quantization (official Qwen release)
**Context:** 65,536 tokens
**Speed:** ~50 tokens/s (WSL2 on Windows)
**Date:** 2026-01-21

### Docker Run Command
```powershell
docker run -d --gpus all `
  --name vllm-multimodal `
  -p 1234:8000 `
  vllm/vllm-openai:latest `
  --model Qwen/Qwen3-VL-8B-Instruct-FP8 `
  --trust-remote-code `
  --max-model-len 65536 `
  --served-model-name qwen3-multimodal `
  --gpu-memory-utilization 0.75 `
  --enable-auto-tool-choice `
  --tool-call-parser hermes
```

---

## Rollback Config (30B MoE)

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

## Alternative: AWQ 4-bit (SLOW - NOT RECOMMENDED)

**Model:** `cyankiwi/Qwen3-VL-8B-Instruct-AWQ-4bit`
**Issue:** Only 3-6 tokens/s on 5090 - Marlin kernels may not be optimized for Blackwell yet

```powershell
# NOT RECOMMENDED - very slow on 5090
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

---

## Performance Comparison

| Config | Model | Quant | Context | Speed | VRAM |
|--------|-------|-------|---------|-------|------|
| FP8 (current) | 8B Dense | FP8 | 64K | ~50 tok/s | ~25GB |
| AWQ 4-bit | 8B Dense | AWQ | 64K | 3-6 tok/s | ~25GB |
| 30B MoE | 30B (3B active) | AWQ | 32K | ~4 tok/s | ~24GB |

**Note:** Speed is limited by WSL2 overhead on Windows. Native Linux would be 2-3x faster.

---

## Useful Commands

```powershell
# Stop and remove container
docker stop vllm-multimodal
docker rm vllm-multimodal

# View logs
docker logs vllm-multimodal

# Follow logs in real-time
docker logs -f vllm-multimodal

# Check throughput (last 10 lines)
docker logs --tail 10 vllm-multimodal

# Test generation speed
Invoke-RestMethod -Uri "http://localhost:1234/v1/completions" -Method Post -ContentType "application/json" -Body '{"model": "qwen3-multimodal", "prompt": "Count from 1 to 100:", "max_tokens": 500}'
```

---

## Context Length Options

If VRAM is tight, reduce `--max-model-len`:
- 65536 (64K) - current
- 49152 (48K)
- 32768 (32K)
- 16384 (16K)
