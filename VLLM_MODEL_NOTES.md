# vLLM Model Configuration Notes

## Current Setup (30B MoE - RECOMMENDED)

**Model:** `QuantTrio/Qwen3-VL-30B-A3B-Instruct-AWQ`
**Type:** 30B MoE (Mixture of Experts) with AWQ quantization (~3B active params)
**Context:** 65,536 tokens
**GPU Utilization:** 60%
**Speed:** ~180 tokens/s
**Date:** 2026-01-21

### Docker Run Command
```powershell
docker run -d --gpus all `
  --name vllm-multimodal `
  -p 1234:8000 `
  vllm/vllm-openai:latest `
  --model QuantTrio/Qwen3-VL-30B-A3B-Instruct-AWQ `
  --trust-remote-code `
  --max-model-len 65536 `
  --served-model-name qwen3-multimodal `
  --gpu-memory-utilization 0.6 `
  --enable-auto-tool-choice `
  --tool-call-parser hermes
```

---

## Alternative: 8B FP8 (slower but more context)

**Model:** `Qwen/Qwen3-VL-8B-Instruct-FP8`
**Type:** 8B Dense with FP8 quantization
**Context:** 65,536 tokens
**Speed:** ~50 tokens/s (limited by WSL2)

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
| **30B MoE (recommended)** | 30B (3B active) | AWQ | 64K | **~180 tok/s** | 60% (~19GB) |
| 8B FP8 | 8B Dense | FP8 | 64K | ~50 tok/s | ~25GB |
| 8B AWQ 4-bit | 8B Dense | AWQ | 64K | 3-6 tok/s | ~25GB |

**Why is MoE faster?** The 30B MoE only activates ~3B parameters per token, making it faster than dense 8B models while maintaining quality of a larger model.

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
