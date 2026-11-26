<p align="center">
  <a href="https://airis-bench.github.io/">
    <img src="https://airis-bench.github.io/assets/logo.png" alt="AIRIS logo" width="150">
  </a>
</p>

<h1 align="center">AIRIS: Benchmarking Social Embodied Intelligence in Egocentric Streaming Videos</h1>

[![Project Page](https://img.shields.io/badge/Project-Page-5B5BD6)](https://airis-bench.github.io/)
[![arXiv](https://img.shields.io/badge/arXiv-2611.34347-b31b1b.svg)](https://arxiv.org/abs/2611.34347)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](#installation)
[![Code License](https://img.shields.io/badge/Code-MIT-green.svg)](LICENSE)
[![Data License](https://img.shields.io/badge/Data-CC%20BY%204.0-lightgrey.svg)](DATA_LICENSE.md)

AIRIS is a causal, time-anchored streaming benchmark for **social embodied intelligence**. It evaluates whether a model can perceive social cues, infer intent only when the evidence is sufficient, choose a timely embodied response, and assess the resulting interaction—without seeing future observations.

> **TL;DR:** AIRIS contains 8,000 expert-curated questions grounded in 357 egocentric clips. It diagnoses social reasoning through a shared four-stage loop: **Attention, Intention, Reaction, and Interaction (A-I-R-I)**.

<p align="center">
  <a href="https://airis-bench.github.io/">
    <img src="https://airis-bench.github.io/assets/figs/fig_teaser.png" alt="AIRIS benchmark overview" width="100%">
  </a>
</p>

## Highlights

- **Causal streaming evaluation:** every query at time $\tau$ receives only the observation prefix $O_{\leq\tau}$.
- **Four-stage diagnosis:** 2,000 questions each for Attention, Intention, Reaction, and Interaction.
- **Broad capability coverage:** 36 dimensions across cue grounding, intent decidability, ego-action selection, task completion, and social-norm compliance.
- **Real-world egocentric interactions:** 357 clips from 276 source videos, totaling 10.91 hours across 11 contexts and 230 fine-grained subscenes.
- **Decision-aware metrics:** AIRIS rewards calibrated waiting and complete reactions while penalizing premature intent commitment and incomplete action chains.
- **Extensive model landscape:** the accompanying paper evaluates 45 proprietary and open-source general-purpose, embodied, streaming-video, and omni-modal models.

## The A-I-R-I loop

AIRIS aligns each social episode to four temporal anchors: $T_0$ (social-cue onset), $T_1$ (intent decidability), $T_2$ (ego-response onset), and $T_3$ (interaction resolution).

1. **Attention** — localize the interaction partner and ground observable gaze, gesture, posture, emotion, and speech cues.
2. **Intention** — infer the partner's latent goal, while selecting **insufficient evidence** before the goal becomes decidable at $T_1$.
3. **Reaction** — decide whether Ego should act, communicate, wait, or refrain, then select the action domain and concrete sub-action.
4. **Interaction** — track the exchange and assess completion, mutual understanding, and social-norm compliance.

See the [project page](https://airis-bench.github.io/) for the interactive capability ontology, benchmark visualizations, qualitative cases, and full model results.

## Benchmark at a glance

| Statistic | Value |
|---|---:|
| Questions | 8,000 |
| A-I-R-I layers | 4 × 2,000 questions |
| Capability dimensions | 36 |
| Egocentric clips | 357 |
| Source videos | 276 |
| Total video duration | 10.91 hours |
| Interaction contexts | 11 |
| Referenced frame timestamps | 20,295 |
| Languages | English and Chinese |

Questions use single-choice, multi-choice, yes/no, and candidate bounding-box formats. Gold answers are public in `data/questions.jsonl`; use `data/model_inputs.jsonl`, which excludes answers, for inference. This release supports reproducible evaluation rather than a hidden-answer challenge.

## Open-release contents

```text
data/
  questions.jsonl              # canonical questions with gold answers
  model_inputs.jsonl           # answer-free inference payload
  clip_metadata.jsonl          # clip-to-source mappings
  source_videos.jsonl          # YouTube IDs and canonical URLs
  frame_index.jsonl            # exact output paths and source timestamps
  annotations/                 # layer-specific annotation sidecars
  audits/                      # selection and hard-negative provenance
predictions/                   # simple public baselines
reports/                       # baseline and reproduction reports
results/                       # public leaderboard summaries
tools/
  fetch_and_extract_frames.py  # local media reconstruction
  streaming_prompt.py          # model-agnostic prompt and frame helpers
  score_streaming.py           # official scorer
  validate_release.py          # release-integrity validator
```

The repository includes annotations, evaluation code, source references, and a local reconstruction tool. It **does not redistribute source videos, audio, or extracted frames**. Locally generated `frames/`, downloaded media, browser cookies, and caches must not be committed.

For field definitions and intended use, see [SCHEMA.md](SCHEMA.md) and [DATA_CARD.md](DATA_CARD.md).

## Installation

Python 3.10 or later is recommended. PyAV also requires FFmpeg libraries; install FFmpeg through your operating system or use a compatible PyAV wheel.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

## Reconstruct the visual inputs

First inspect the complete media plan offline:

```bash
python3 tools/fetch_and_extract_frames.py --stage plan
```

The expected plan contains 276 source videos and 20,295 unique frames. To download accessible source videos and extract the referenced frames:

```bash
python3 tools/fetch_and_extract_frames.py --stage all --jobs 2
```

The script downloads each source once into `.cache/youtube/`, extracts the nearest decoded frame for each released timestamp, writes the expected paths under `frames/`, verifies image dimensions, and records the build in `reports/media_build.json`.

Only access or download media when you have the right to do so. You are responsible for complying with YouTube's Terms of Service, uploader licenses, copyright, privacy, publicity rights, and applicable law. Public availability does not grant permission to redistribute source media or reconstructed frames. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Prepare model inputs

Each record in `data/model_inputs.jsonl` contains the question, options, causal frame sequence, and admissible speech context. `tools/streaming_prompt.py` provides a reference adapter:

```python
import json
from pathlib import Path

from tools.streaming_prompt import build_prompt, load_visual_inputs

root = Path(".")
with (root / "data/model_inputs.jsonl").open(encoding="utf-8") as f:
    question = json.loads(next(f))

prompt = build_prompt(question, language="en")
images = load_visual_inputs(question, root)
```

For bounding-box questions, the helper overlays labeled candidate boxes on the final causal frame. Submit option keys only; multi-choice predictions may contain multiple keys.

## Evaluate predictions

Write one JSON object per line:

```json
{"qid": "pub.stream.example", "pred": "A"}
{"qid": "pub.stream.multi_example", "pred": ["A", "C"]}
```

Run the official scorer:

```bash
python3 tools/score_streaming.py \
  --questions data/questions.jsonl \
  --predictions predictions.jsonl \
  --output report.json
```

Single-choice, yes/no, and candidate-box items use exact match; multi-choice items use set F1. Missing, empty, invalid, and out-of-vocabulary predictions receive zero credit on the fixed 8,000-question denominator.

The headline score is the unweighted mean of the four question-micro layer scores:

$$
\mathrm{Overall} =
\frac{S_{\mathrm{Attention}} + S_{\mathrm{Intention}} +
S_{\mathrm{Reaction}} + S_{\mathrm{Interaction}}}{4}.
$$

The report also includes stage-specific diagnostics such as semantic/spatial grounding, False Commitment Rate and Intent Anticipation Score, engagement macro-F1, reaction Joint-EM, and interaction compliance and violation metrics.

## Validate the release

Validate indexes, schemas, counts, answer contracts, and packaged-file checksums:

```bash
python3 tools/validate_release.py
```

After reconstructing the media, additionally verify every required frame:

```bash
python3 tools/validate_release.py --require-frames
```

## Reproduction limits

Released frame mappings follow:

```text
source_time_ms = source_video.start_sec * 1000 + clip_time_ms
```

Source videos may become unavailable, region-restricted, edited, replaced, or served through different transcodes. The original download formats and JPEG encoder settings were not retained. Reconstruction therefore targets the released semantic inputs, timestamps, and dimensions—not byte-identical historical JPEG files.

## Results

Public result summaries and machine-readable scores are available in [`results/`](results/). The paper's central findings are:

- leading general-purpose VLMs reach 93.6% on Attention, but strong perception does not reliably transfer to embodied decision-making;
- model families prematurely commit to an intent on 26%–51% of pre-$T_1$ queries;
- no evaluated model exceeds 14.7% Joint-EM when reaction timing, action domain, concrete execution, and applicable speech function must all be correct together.

See the [interactive leaderboard](https://airis-bench.github.io/#leaderboard) for all 45 evaluated models and detailed diagnostic views.

## Licenses and responsible use

- Source code in `tools/` is released under the [MIT License](LICENSE).
- AIRIS-authored annotations, metadata, benchmark summaries, and documentation are released under [CC BY 4.0](DATA_LICENSE.md).
- Third-party source media and locally reconstructed frames are not distributed and are not covered by either license.

AIRIS is intended for research on causal streaming multimodal understanding and social embodied intelligence. Identity recognition, biometric profiling, surveillance, sensitive-trait inference, and safety-critical or high-impact deployment are out of scope. Consult the [Dataset Card](DATA_CARD.md) before use.

## Citation

If AIRIS is useful in your research, please cite the paper:

```bibtex
@article{zhou2026airis,
  title         = {AIRIS: Benchmarking Social Embodied Intelligence in Egocentric Streaming Videos},
  author        = {Zhou, Yuchen and Wu, Hongsheng and Tang, Jiayu and Mao, Yunxuan and Ye, Hanjing and Zhang, Shenghao and Tang, Jing and Fan, Chun-Kai and Huang, Yufei and Ling, Yonggen and Zhang, He},
  journal       = {arXiv preprint arXiv:2611.34347},
  year          = {2026},
  eprint        = {2611.34347},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV}
}
```

Dataset citation metadata is also provided in [`CITATION.cff`](CITATION.cff).

## Corrections and takedown requests

Please open an issue for benchmark corrections, source availability, privacy concerns, or takedown requests. For source-media requests, include the YouTube video ID and affected `clip_id` values, but do not post private information.
