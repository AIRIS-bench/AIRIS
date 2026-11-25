# Data Schema

All JSONL files contain one UTF-8 JSON object per line.

## `data/questions.jsonl`

Canonical gold questions. Important fields:

- `qid`: stable question identifier.
- `capability`: capability name, prefixed by layer.
- `qtype`: `single_choice`, `multi_choice`, `yes_no`, or `bbox_select`.
- `question`, `question_zh`: English and Chinese question text.
- `frames`: causally visible frames in chronological order.
- `options`: keyed answer candidates; bounding-box options use normalized coordinates.
- `answer`: gold key or list of keys.
- `extra`: task-specific temporal and annotation metadata. Internal reviewer and LLM-operation fields were removed.

## `data/model_inputs.jsonl`

Inference payload with the same `qid`, question, options, and frame contract. Gold answers, reviewer metadata, option-compression metadata, and levels are excluded.

## Frame entries

```json
{
  "clip_id": "...",
  "t_ms": 7300,
  "frame_path": "frames/<clip_id>/<clip_id>__000007300.jpg",
  "width": 1080,
  "height": 1920
}
```

`t_ms` is clip-local time in milliseconds, not a frame number.

## `data/clip_metadata.jsonl`

One row per benchmark clip:

- `clip_id`: stable clip identifier.
- `source_video.platform`: `youtube`.
- `source_video.source_id`: legacy `watch-...` or `shorts-...` identifier.
- `source_video.video_id`: exact 11-character YouTube video ID. Leading `-` or `_` is significant.
- `source_video.url`: canonical watch URL.
- `source_video.start_sec`: source-video offset used by frame reconstruction.
- `source_video.end_sec`: legacy nominal end; informational only.
- `duration_ms`, `fps`, `width`, `height`: benchmark clip metadata.

Do not parse a video ID with a general split or strip operation. Remove only the first fixed `watch-` or `shorts-` prefix.

## `data/source_videos.jsonl`

One row per unique YouTube video. It contains the canonical URL, exact video ID, expected dimensions, related clip IDs, and an availability field. Availability is `not_checked` in the static release and must not be interpreted as confirmation that a link is currently accessible.

## `data/frame_index.jsonl`

One row per unique required output frame:

- `frame_path`: required repository-relative JPEG path.
- `clip_id`, `clip_time_ms`: clip-local mapping.
- `source_video_id`, `youtube_url`: source reference.
- `source_time_ms`: absolute source-video timestamp used by the extraction script.
- `width`, `height`: required output dimensions.
- `mapping_status`: provenance of the timestamp mapping.

The mapping is:

```text
source_time_ms = source_video.start_sec * 1000 + clip_time_ms
```

The extraction script uses `source_time_ms` directly; it does not infer duration from `end_sec`.

## Prediction schema

Each prediction is a JSONL object:

```json
{"qid":"<question id>","pred":"A"}
```

For `multi_choice`, `pred` may be a list such as `["A", "C"]`. Unknown, missing, empty, or invalid predictions are scored as incorrect.
