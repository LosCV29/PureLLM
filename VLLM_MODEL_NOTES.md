# vLLM Model Configuration Notes

## Current Setup (30B MoE - RECOMMENDED)

**Model:** `QuantTrio/Qwen3-VL-30B-A3B-Instruct-AWQ`
**Type:** 30B MoE (Mixture of Experts) with AWQ quantization (~3B active params)
**Context:** 32,768 tokens
**GPU Utilization:** 70%
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
  --max-model-len 32768 `
  --served-model-name qwen3-multimodal `
  --gpu-memory-utilization 0.7 `
  --enable-auto-tool-choice `
  --tool-call-parser hermes
```

---

## Alternative: 8B FP8 (more context, slower)

**Model:** `Qwen/Qwen3-VL-8B-Instruct-FP8`
**Type:** 8B Dense with FP8 quantization
**Context:** 65,536 tokens
**Speed:** ~50 tokens/s (limited by WSL2)

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
**Issue:** Only 3-6 tokens/s on 5090 - Marlin kernels not optimized for Blackwell

---

## Performance Comparison (RTX 5090 + WSL2)

| Config | Model | Quant | Context | Speed | Notes |
|--------|-------|-------|---------|-------|-------|
| **30B MoE (recommended)** | 30B (3B active) | AWQ | 32K | **~180 tok/s** | Best speed |
| 8B FP8 | 8B Dense | FP8 | 64K | ~50 tok/s | More context |
| 8B AWQ 4-bit | 8B Dense | AWQ | 64K | 3-6 tok/s | Broken on 5090 |

**Why is MoE faster?** Only activates ~3B parameters per token despite being a "30B" model.

**Why 64K fails on MoE?** The 30B model uses ~17GB VRAM. With 70% utilization (~23GB of 32GB), there's only ~6GB for KV cache. 64K context needs 6GB KV cache minimum, leaving no headroom. 32K works.

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

# Test generation speed
Invoke-RestMethod -Uri "http://localhost:1234/v1/completions" -Method Post -ContentType "application/json" -Body '{"model": "qwen3-multimodal", "prompt": "Count from 1 to 100:", "max_tokens": 500}'
```
