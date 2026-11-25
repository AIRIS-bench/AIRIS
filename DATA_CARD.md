# Dataset Card for AIRIS

## Summary

AIRIS Streaming Benchmark v02.5-core3 is a test-only benchmark for causal streaming understanding of social interactions. It evaluates social embodied intelligence as a time-anchored chain of **Attention, Intention, Reaction, and Interaction (A-I-R-I)**. Every query is evaluated using only the observation prefix available at its designated timestamp.

## Data composition

- Questions: 8,000 bilingual English/Chinese items
- A-I-R-I layers: 2,000 questions per layer
- Capabilities: 36 (19 Attention, 3 Intention, 4 Reaction, and 10 Interaction)
- Egocentric clips: 357
- Unique YouTube source videos: 276
- Total source-clip duration: 10.91 hours
- Interaction contexts: 11
- Fine-grained subscenes: 230
- Unique referenced frame timestamps: 20,295
- Question types: single choice, multiple choice, yes/no, and bounding-box selection

The release includes AIRIS-authored questions, annotations, source references, evaluation code, and public aggregate results. It excludes source videos, audio, thumbnails, and frame images. Gold answers are public, so this release supports reproducible evaluation rather than a hidden-answer challenge.

## Task formulation

AIRIS aligns each episode to four temporal anchors: social-cue onset (`T0`), intent decidability (`T1`), ego-response onset (`T2`), and interaction resolution (`T3`).

- **Attention** grounds the interaction partner and observable social cues.
- **Intention** recovers the partner's latent goal and rewards calibrated waiting before `T1`.
- **Reaction** evaluates whether and how the ego-agent should respond.
- **Interaction** assesses episode state, completion, mutual understanding, and social-norm compliance.

The headline score is the unweighted mean of the four question-micro layer scores. Single-choice, yes/no, and candidate-box items use exact match; multi-choice items use set F1. The scorer also reports stage-specific diagnostics for grounding, premature commitment, intent anticipation, engagement, joint reaction correctness, and interaction compliance.

## Media acquisition

`data/source_videos.jsonl` contains canonical YouTube links and video IDs. `data/frame_index.jsonl` contains the source timestamp for every required frame. Users generate media locally with `tools/fetch_and_extract_frames.py`.

Video availability is not guaranteed. The release does not grant rights to source videos and does not bypass access controls or digital rights management. Public availability is not permission to redistribute source media or reconstructed frames.

## Intended uses

- Research on streaming multimodal understanding.
- Evaluation of causal visual-language reasoning.
- Analysis of attention, intention, reaction, and interaction capabilities.
- Reproducible comparison with the included evaluation protocol.

## Out-of-scope uses

- Identity recognition, biometric profiling, surveillance, or tracking.
- Inferring sensitive traits about depicted people.
- Deployment in safety-critical or high-impact decisions.
- Redistribution of downloaded videos or frames without permission.

## Privacy, safety, and representation

Source videos may depict identifiable adults, children, workers, public figures, payment scenes, or people in sensitive contexts. Speech context may contain names or other information spoken in public videos. This release removes internal reviewer identities and internal filesystem paths, but it does not claim that all third-party personal information has been removed from source content.

Users should perform an independent rights and privacy review for their jurisdiction and use case. Do not publish reconstructed media merely because a source URL is public.

## Known limitations

- Test-only release; no train/validation split is provided.
- Gold answers are public.
- YouTube links may become unavailable or change over time.
- Historical video download formats and JPEG encoding settings were not retained.
- Frame reconstruction follows released timestamps but is not guaranteed to be byte-identical to historical frames.
- Current YouTube transcodes may have different resolution, color conversion, timing, or edits.
- The benchmark uses sampled visual frames and time-aligned speech context for standard VLM evaluation; it does not reproduce every model's native streaming interface.
- Capability and scene distributions are not representative of all cultures or real-world interactions.

## Maintenance

Submit corrections and takedown requests through the issue tracker of the repository hosting this release. Requests concerning source rights should identify the YouTube video ID and affected `clip_id` values without posting private information. Maintainers should disable affected mappings while reviewing rights, privacy, availability, or annotation concerns. Link availability and privacy review are ongoing maintenance activities rather than guarantees of this release.
