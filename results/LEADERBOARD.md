# AIRIS Leaderboard

Zero-shot results on the 8,000-question AIRIS Streaming Benchmark v02.5-core3. Scores are in `[0, 1]` (higher is better). Overall is the unweighted mean of the four question-micro A-I-R-I layer scores. Missing, empty, invalid, and out-of-vocabulary predictions receive zero credit.

The table contains all 45 models evaluated in the paper; 37 appear in the main-paper table. See [`leaderboard.json`](leaderboard.json) for stage-specific diagnostics and the [interactive leaderboard](https://airis-bench.github.io/#leaderboard) for sortable views.

| Rank | Model | Family | Overall | Attention | Intention | Reaction | Interaction |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | [Gemini-3.7-Flash](https://ai.google.dev/gemini-api/docs/models/gemini-3.7-flash) | Closed-source | 0.726 | 0.936 | 0.569 | 0.638 | 0.760 |
| 2 | [Gemini-3.1-Pro](https://ai.google.dev/gemini-api/docs/models/gemini-3.1-pro-preview) | Closed-source | 0.682 | 0.851 | 0.496 | 0.620 | 0.760 |
| 3 | [GPT-5.6-Sol](https://developers.openai.com/api/docs/models/gpt-5.6-sol) | Closed-source | 0.656 | 0.906 | 0.501 | 0.489 | 0.729 |
| 4 | [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) | Open general | 0.650 | 0.723 | 0.485 | 0.655 | 0.738 |
| 5 | [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) | Open general | 0.633 | 0.801 | 0.501 | 0.590 | 0.638 |
| 6 | [Gemma-4-31B-IT](https://huggingface.co/google/gemma-4-31B-it) | Open general | 0.624 | 0.733 | 0.502 | 0.535 | 0.728 |
| 7 | [Doubao-seed-2.0-lite](https://seed.bytedance.com/zh/seed2) | Closed-source | 0.616 | 0.759 | 0.418 | 0.619 | 0.669 |
| 8 | [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) | Open general | 0.605 | 0.707 | 0.509 | 0.548 | 0.657 |
| 9 | [Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) | Open general | 0.602 | 0.649 | 0.480 | 0.545 | 0.733 |
| 10 | [HY-Embodied-VLM-1.0](https://huggingface.co/tencent/Hy-Embodied-VLM-1.0) | Open embodied | 0.599 | 0.596 | 0.396 | 0.688 | 0.716 |
| 11 | [Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) | Open general | 0.595 | 0.608 | 0.448 | 0.609 | 0.715 |
| 12 | [Qwen3-VL-32B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct) | Open general | 0.589 | 0.682 | 0.472 | 0.479 | 0.723 |
| 13 | [RynnBrain1.1-9B](https://huggingface.co/Alibaba-DAMO-Academy/RynnBrain1.1-9B) | Open embodied | 0.576 | 0.593 | 0.417 | 0.557 | 0.736 |
| 14 | [JoyAI-VL-Interaction-Preview](https://huggingface.co/jdopensource/JoyAI-VL-Interaction-Preview) | Open streaming | 0.568 | 0.639 | 0.436 | 0.610 | 0.586 |
| 15 | [Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B) | Open general | 0.561 | 0.668 | 0.382 | 0.539 | 0.656 |
| 16 | [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | Open general | 0.547 | 0.685 | 0.395 | 0.475 | 0.632 |
| 17 | [RynnBrain-30B-A3B](https://huggingface.co/Alibaba-DAMO-Academy/RynnBrain-30B-A3B) | Open embodied | 0.544 | 0.656 | 0.435 | 0.410 | 0.674 |
| 18 | [VeBrain-7B](https://huggingface.co/OpenGVLab/VeBrain) | Open embodied | 0.541 | 0.546 | 0.449 | 0.470 | 0.701 |
| 19 | [Pelican1.0-VL-7B](https://huggingface.co/X-Humanoid/Pelican1.0-VL-7B) | Open embodied | 0.536 | 0.540 | 0.462 | 0.440 | 0.700 |
| 20 | [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) | Open general | 0.535 | 0.507 | 0.452 | 0.466 | 0.713 |
| 21 | [AURA-8B](https://huggingface.co/aurateam/AURA) | Open streaming | 0.533 | 0.624 | 0.472 | 0.434 | 0.601 |
| 22 | [InternVL3-8B](https://huggingface.co/OpenGVLab/InternVL3-8B) | Open general | 0.530 | 0.583 | 0.417 | 0.518 | 0.603 |
| 23 | [Qwen2.5-Omni-7B](https://huggingface.co/Qwen/Qwen2.5-Omni-7B) | Open omni-modal | 0.529 | 0.507 | 0.453 | 0.449 | 0.705 |
| 24 | [InternVL3.5-30B-A3B](https://huggingface.co/OpenGVLab/InternVL3_5-30B-A3B) | Open general | 0.526 | 0.598 | 0.426 | 0.442 | 0.639 |
| 25 | [MiMo-Embodied-7B](https://huggingface.co/XiaomiMiMo/MiMo-Embodied-7B) | Open embodied | 0.526 | 0.533 | 0.440 | 0.643 | 0.487 |
| 26 | [Qwen3-Omni-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct) | Open omni-modal | 0.526 | 0.646 | 0.419 | 0.390 | 0.647 |
| 27 | [StreamingVLM](https://huggingface.co/mit-han-lab/StreamingVLM) | Open streaming | 0.512 | 0.456 | 0.400 | 0.536 | 0.655 |
| 28 | [RynnBrain1.1-2B](https://huggingface.co/Alibaba-DAMO-Academy/RynnBrain1.1-2B) | Open embodied | 0.508 | 0.602 | 0.396 | 0.467 | 0.567 |
| 29 | [RoboBrain2.0-7B](https://huggingface.co/BAAI/RoboBrain2.0-7B) | Open embodied | 0.505 | 0.511 | 0.439 | 0.471 | 0.597 |
| 30 | [InternVL3.5-8B](https://huggingface.co/OpenGVLab/InternVL3_5-8B) | Open general | 0.500 | 0.574 | 0.406 | 0.458 | 0.562 |
| 31 | [RoboBrain2.0-32B](https://huggingface.co/BAAI/RoboBrain2.0-32B) | Open embodied | 0.487 | 0.486 | 0.451 | 0.441 | 0.571 |
| 32 | [OmniVinci](https://huggingface.co/nvidia/omnivinci) | Open omni-modal | 0.484 | 0.546 | 0.401 | 0.340 | 0.651 |
| 33 | [Embodied-R1-3B-V1](https://huggingface.co/IffYuan/Embodied-R1-3B-v1) | Open embodied | 0.480 | 0.475 | 0.408 | 0.468 | 0.569 |
| 34 | [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) | Open general | 0.476 | 0.495 | 0.409 | 0.483 | 0.515 |
| 35 | [Qwen2.5-Omni-3B](https://huggingface.co/Qwen/Qwen2.5-Omni-3B) | Open omni-modal | 0.474 | 0.477 | 0.433 | 0.467 | 0.520 |
| 36 | [Flash-VStream-Qwen-7B](https://huggingface.co/zhang9302002/Flash-VStream-Qwen-7b) | Open streaming | 0.468 | 0.470 | 0.396 | 0.376 | 0.631 |
| 37 | [LLaVA-OneVision-7B](https://huggingface.co/llava-hf/llava-onevision-qwen2-7b-ov-hf) | Open general | 0.467 | 0.500 | 0.387 | 0.379 | 0.602 |
| 38 | [HY-Embodied-0.5](https://huggingface.co/tencent/HY-Embodied-0.5) | Open embodied | 0.461 | 0.482 | 0.368 | 0.563 | 0.429 |
| 39 | [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) | Open general | 0.457 | 0.534 | 0.389 | 0.481 | 0.425 |
| 40 | [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) | Open general | 0.445 | 0.477 | 0.375 | 0.530 | 0.396 |
| 41 | [VITA-1.5](https://huggingface.co/VITA-MLLM/VITA-1.5) | Open streaming | 0.436 | 0.460 | 0.388 | 0.420 | 0.477 |
| 42 | [HY-Embodied-0.5-X](https://huggingface.co/tencent/HY-Embodied-0.5) | Open embodied | 0.432 | 0.461 | 0.373 | 0.480 | 0.415 |
| 43 | [Llama-3.2-11B-Vision-Instruct](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct) | Open general | 0.423 | 0.391 | 0.384 | 0.463 | 0.455 |
| 44 | [Phi-4-Multimodal-Instruct](https://huggingface.co/microsoft/Phi-4-multimodal-instruct) | Open omni-modal | 0.406 | 0.435 | 0.398 | 0.435 | 0.356 |
| 45 | [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) | Open general | 0.396 | 0.381 | 0.299 | 0.506 | 0.398 |

## Metric notes

- **Attention:** question-micro score over observable social-cue and spatial-grounding tasks.
- **Intention:** question-micro score over key-moment, temporal, and evidence-grounding tasks.
- **Reaction:** question-micro score over engagement, action-domain, sub-action, and speech-function tasks.
- **Interaction:** question-micro score over episode state, outcome, and social-norm compliance tasks.
- Multi-choice questions use set F1; single-choice, yes/no, and candidate-box questions use exact match.
