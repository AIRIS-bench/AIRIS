#!/usr/bin/env python3
"""Validate the standalone AIRIS open release without parent datasets."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
MODEL_INPUT_FORBIDDEN_KEYS = {
    "answer",
    "gold_key",
    "human_review",
    "level",
    "option_compression",
}
RUNTIME_DIRS = {".cache", ".git", "__pycache__", "frames", "reports"}
UNCHECKSUMMED_FILES = {"manifest.json", "validation_report.json"}
FORBIDDEN_RELEASE_SUFFIXES = {
    ".avi",
    ".jpeg",
    ".jpg",
    ".m4a",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp3",
    ".mp4",
    ".png",
    ".wav",
    ".webm",
}
ANNOTATION_FILES = {
    "data/annotations/intention.jsonl": "intention",
    "data/annotations/reaction.jsonl": "reaction",
    "data/annotations/interaction.jsonl": "interaction",
    "data/annotations/interaction_v2.jsonl": "interaction",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def nested_keys(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key)
            yield from nested_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from nested_keys(child)


def model_payload(question: Mapping[str, Any]) -> dict[str, Any]:
    extra = question.get("extra") or {}
    return {
        "qid": question["qid"],
        "capability": question["capability"],
        "qtype": question["qtype"],
        "question": question["question"],
        "question_zh": question.get("question_zh"),
        "language": question.get("language", "en"),
        "frames": question.get("frames") or [],
        "options": [
            {key: option[key] for key in ("key", "text", "text_zh", "bbox", "target") if key in option}
            for option in question.get("options") or []
        ],
        "extra": {
            key: extra[key]
            for key in ("speech_context", "speech_context_original", "speech_context_en")
            if extra.get(key)
        },
    }


def expected_frame_path(clip_id: str, t_ms: int) -> str:
    return f"frames/{clip_id}/{clip_id}__{t_ms:09d}.jpg"


def safe_relative_path(value: str, *, first_part: str | None = None, parts: int | None = None) -> bool:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        return False
    if first_part is not None and path.parts[0] != first_part:
        return False
    return parts is None or len(path.parts) == parts


def has_symlink_component(root: Path, relative: str) -> bool:
    current = root
    for part in PurePosixPath(relative).parts:
        current = current / part
        if current.is_symlink():
            return True
    return False


def scan_release_tree(root: Path, checksums: Mapping[str, str]) -> list[str]:
    failures: list[str] = []
    tracked_files: set[str] = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        relative_text = relative.as_posix()
        if any(part == ".git" for part in relative.parts):
            continue
        if path.is_symlink():
            failures.append(f"release tree contains symlink: {relative_text}")
            continue
        if not path.is_file():
            continue
        if any(part in RUNTIME_DIRS for part in relative.parts):
            continue
        if path.suffix.lower() in FORBIDDEN_RELEASE_SUFFIXES:
            failures.append(f"release tree contains media file: {relative_text}")
        if relative_text not in UNCHECKSUMMED_FILES:
            tracked_files.add(relative_text)
    checksum_files = set(checksums)
    for relative in sorted(tracked_files - checksum_files):
        failures.append(f"file is not covered by manifest checksum: {relative}")
    for relative in sorted(checksum_files - tracked_files):
        failures.append(f"manifest checksum references missing/untracked file: {relative}")
    return failures


def validate(root: Path, require_frames: bool) -> dict[str, Any]:
    failures: list[str] = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            failures.append(message)

    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    questions = read_jsonl(root / "data/questions.jsonl")
    inputs = read_jsonl(root / "data/model_inputs.jsonl")
    clips = read_jsonl(root / "data/clip_metadata.jsonl")
    sources = read_jsonl(root / "data/source_videos.jsonl")
    frame_index = read_jsonl(root / "data/frame_index.jsonl")

    expected = manifest["counts"]
    check(len(questions) == expected["questions"], f"questions: {len(questions)} != {expected['questions']}")
    check(len(inputs) == expected["questions"], f"model_inputs: {len(inputs)} != {expected['questions']}")
    check(len(clips) == expected["clips"], f"clips: {len(clips)} != {expected['clips']}")
    check(len(sources) == expected["source_videos"], f"sources: {len(sources)} != {expected['source_videos']}")
    check(len(frame_index) == expected["referenced_frames"], f"frame_index: {len(frame_index)} != {expected['referenced_frames']}")

    qids = [str(row.get("qid")) for row in questions]
    input_qids = [str(row.get("qid")) for row in inputs]
    check(len(qids) == len(set(qids)), "questions contain duplicate qids")
    check(qids == input_qids, "questions and model_inputs qid order differs")
    question_map = {str(row["qid"]): row for row in questions}

    layer_counts = Counter(str(row.get("capability", "")).split(".", 1)[0] for row in questions)
    capability_count = len({str(row.get("capability", "")) for row in questions})
    check(dict(sorted(layer_counts.items())) == manifest["counts"]["by_layer"], "layer counts differ from manifest")
    check(capability_count == expected["capabilities"], "capability count differs from manifest")

    for row in questions:
        keys = {str(option.get("key")) for option in row.get("options") or []}
        answer = row.get("answer")
        answers = answer if isinstance(answer, list) else [answer]
        if not answers or any(str(item) not in keys for item in answers):
            failures.append(f"{row.get('qid')}: answer is outside option keys")
        extra = row.get("extra") or {}
        if "human_review" in extra or any(str(key).startswith("llm_") for key in extra):
            failures.append(f"{row.get('qid')}: internal review metadata was not removed")

    for question, model_input in zip(questions, inputs):
        leaked = MODEL_INPUT_FORBIDDEN_KEYS.intersection(nested_keys(model_input))
        if leaked:
            failures.append(f"{model_input.get('qid')}: model input leaks {sorted(leaked)}")
        if model_input != model_payload(question):
            failures.append(f"{question.get('qid')}: model input is not the canonical gold-free projection")

    annotation_qids: dict[str, set[str]] = defaultdict(set)
    for relative, layer in ANNOTATION_FILES.items():
        rows = read_jsonl(root / relative)
        local_qids: set[str] = set()
        for row in rows:
            qid = str(row.get("qid"))
            check(qid not in local_qids, f"{relative}: duplicate qid {qid}")
            local_qids.add(qid)
            annotation_qids[layer].add(qid)
            check(qid in question_map, f"{relative}: unknown qid {qid}")
            if qid in question_map:
                check(row == question_map[qid], f"{relative}: row differs from questions.jsonl for {qid}")
                check(str(row.get("capability", "")).startswith(layer + "."), f"{relative}: wrong layer for {qid}")
    for layer in ("intention", "interaction", "reaction"):
        expected_qids = {qid for qid, row in question_map.items() if str(row["capability"]).startswith(layer + ".")}
        check(annotation_qids[layer] == expected_qids, f"annotation coverage differs for layer {layer}")

    clip_map = {str(row["clip_id"]): row for row in clips}
    check(len(clip_map) == len(clips), "clip metadata contains duplicate clip_id")
    source_map = {str(row["video_id"]): row for row in sources}
    check(len(source_map) == len(sources), "source_videos contains duplicate video_id")
    clips_by_source: dict[str, set[str]] = defaultdict(set)

    for clip_id, row in clip_map.items():
        source = row.get("source_video") or {}
        video_id = str(source.get("video_id", ""))
        clips_by_source[video_id].add(clip_id)
        check(bool(VIDEO_ID_RE.fullmatch(video_id)), f"{clip_id}: invalid video ID")
        check(source.get("platform") == "youtube", f"{clip_id}: invalid source platform")
        check(source.get("url") == f"https://www.youtube.com/watch?v={video_id}", f"{clip_id}: invalid YouTube URL")
        check("folder" not in source and "file_name" not in source, f"{clip_id}: internal source path remains")
        check(video_id in source_map, f"{clip_id}: source video missing from source_videos.jsonl")

    for video_id, row in source_map.items():
        check(bool(VIDEO_ID_RE.fullmatch(video_id)), f"source list: invalid video ID {video_id!r}")
        check(row.get("platform") == "youtube", f"{video_id}: invalid source platform")
        check(row.get("url") == f"https://www.youtube.com/watch?v={video_id}", f"{video_id}: invalid canonical URL")
        row_clips = {str(value) for value in row.get("clip_ids") or []}
        check(row_clips == clips_by_source[video_id], f"{video_id}: clip_ids differ from clip metadata")
        check(int(row.get("n_clips", -1)) == len(row_clips), f"{video_id}: n_clips is incorrect")
        check(row.get("availability") in {"not_checked", "available", "unavailable"}, f"{video_id}: invalid availability")
        dimensions = {(int(clip_map[clip_id]["width"]), int(clip_map[clip_id]["height"])) for clip_id in row_clips}
        check(dimensions == {(int(row.get("expected_width", 0)), int(row.get("expected_height", 0)))}, f"{video_id}: dimensions differ")

    indexed: dict[str, dict[str, Any]] = {}
    for row in frame_index:
        relative = str(row.get("frame_path", ""))
        clip_id = str(row.get("clip_id", ""))
        t_ms = int(row.get("clip_time_ms", -1))
        check(relative not in indexed, f"duplicate frame index path: {relative}")
        indexed[relative] = row
        check(safe_relative_path(relative, first_part="frames", parts=3), f"unsafe frame path: {relative}")
        check(relative == expected_frame_path(clip_id, t_ms), f"frame path/timestamp mismatch: {relative}")
        clip = clip_map.get(clip_id)
        if clip is None:
            failures.append(f"{relative}: unknown clip")
            continue
        source = clip["source_video"]
        expected_source_ms = int(round(float(source["start_sec"]) * 1000)) + t_ms
        check(int(row.get("source_time_ms", -1)) == expected_source_ms, f"{relative}: source timestamp mismatch")
        check(row.get("source_video_id") == source["video_id"], f"{relative}: source video mismatch")
        check(row.get("youtube_url") == source["url"], f"{relative}: YouTube URL mismatch")
        check((int(row.get("width", 0)), int(row.get("height", 0))) == (int(clip["width"]), int(clip["height"])), f"{relative}: dimensions differ from clip")

    referenced: set[str] = set()
    for question in questions:
        previous = -1
        for frame in question.get("frames") or []:
            clip_id = str(frame.get("clip_id", ""))
            t_ms = int(frame.get("t_ms", -1))
            relative = str(frame.get("frame_path", ""))
            check(t_ms >= previous, f"{question.get('qid')}: frames are not chronological")
            check(relative == expected_frame_path(clip_id, t_ms), f"{question.get('qid')}: malformed frame path")
            check(relative in indexed, f"{question.get('qid')}: frame absent from frame index: {relative}")
            if relative in indexed:
                index_row = indexed[relative]
                check((int(frame.get("width", 0)), int(frame.get("height", 0))) == (int(index_row["width"]), int(index_row["height"])), f"{question.get('qid')}: frame dimensions differ from index")
            previous = t_ms
            referenced.add(relative)
    check(referenced == set(indexed), "frame index and question frame references differ")

    missing_frames: int | None = None
    invalid_frames: int | None = None
    if require_frames:
        missing_frames = 0
        invalid_frames = 0
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("Pillow is required for --require-frames") from exc
        for relative, row in indexed.items():
            path = root.joinpath(*PurePosixPath(relative).parts)
            if has_symlink_component(root, relative) or not path.is_file():
                missing_frames += 1
                continue
            try:
                with Image.open(path) as image:
                    image.verify()
                with Image.open(path) as image:
                    if image.format != "JPEG" or image.size != (int(row["width"]), int(row["height"])):
                        invalid_frames += 1
            except Exception:
                invalid_frames += 1
        check(missing_frames == 0, f"missing frames: {missing_frames}")
        check(invalid_frames == 0, f"invalid frames: {invalid_frames}")

    checksums = manifest.get("checksums", {})
    checksum_failures = 0
    for relative, expected_digest in checksums.items():
        if not safe_relative_path(str(relative)) or not SHA256_RE.fullmatch(str(expected_digest)):
            checksum_failures += 1
            failures.append(f"unsafe or malformed checksum entry: {relative}")
            continue
        path = root.joinpath(*PurePosixPath(str(relative)).parts)
        if has_symlink_component(root, str(relative)) or not path.is_file() or sha256(path) != expected_digest:
            checksum_failures += 1
            failures.append(f"checksum mismatch: {relative}")
    failures.extend(scan_release_tree(root, checksums))

    return {
        "schema_version": "airis-open-release-validation-v1",
        "status": "passed" if not failures else "failed",
        "n_failures": len(failures),
        "failures": failures[:200],
        "failures_truncated": max(0, len(failures) - 200),
        "counts": {
            "questions": len(questions),
            "clips": len(clips),
            "source_videos": len(sources),
            "referenced_frames": len(frame_index),
            "capabilities": capability_count,
            "checksum_failures": checksum_failures,
            "missing_frames": missing_frames,
            "invalid_frames": invalid_frames,
        },
        "media_mode": "required" if require_frames else "metadata_only",
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=default_root)
    parser.add_argument("--require-frames", action="store_true")
    parser.add_argument("--output", default="validation_report.json")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root.resolve()
    report = validate(root, args.require_frames)
    output = Path(args.output)
    if not output.is_absolute():
        output = root / output
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{report['status']}: {report['n_failures']} failure(s); wrote {output}")
    return int(report["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
