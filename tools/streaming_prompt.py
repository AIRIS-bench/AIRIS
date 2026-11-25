"""Model-agnostic input contract for AIRIS Streaming questions."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Any

from PIL import Image, ImageDraw, ImageFont

BBOX_COLORS = ("red", "deepskyblue", "limegreen", "gold", "magenta", "cyan")


def _safe_frame_path(release_dir: Path, relative: str) -> Path:
    path = PurePosixPath(str(relative))
    if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != "frames":
        raise ValueError(f"unsafe frame path: {relative!r}")
    release_dir = release_dir.resolve()
    current = release_dir
    for part in path.parts:
        current = current / part
        if current.exists() and current.is_symlink():
            raise ValueError(f"symlink is not allowed in frame path: {relative!r}")
    target = (release_dir / Path(*path.parts)).resolve()
    frames_root = (release_dir / "frames").resolve()
    if target != frames_root and frames_root not in target.parents:
        raise ValueError(f"frame path escapes release root: {relative!r}")
    if target.is_symlink():
        raise ValueError(f"frame path must not be a symlink: {relative!r}")
    return target


def frame_paths(question: dict[str, Any], release_dir: Path) -> list[Path]:
    """Return causally visible frames in chronological order."""
    paths = [
        _safe_frame_path(release_dir, str(frame.get("frame_path") or ""))
        for frame in question.get("frames") or []
    ]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"{question.get('qid')}: missing frames: {missing[:3]}")
    return paths


def _display_text(question: dict[str, Any], language: str) -> tuple[str, list[tuple[str, str]]]:
    if language == "zh":
        stem = str(question.get("question_zh") or question.get("question") or "")
        options = [
            (str(option.get("key")), str(option.get("text_zh") or option.get("text") or ""))
            for option in question.get("options") or []
        ]
    else:
        stem = str(question.get("question") or "")
        options = [
            (str(option.get("key")), str(option.get("text") or ""))
            for option in question.get("options") or []
        ]
    return stem, options


def build_prompt(question: dict[str, Any], *, language: str = "en") -> str:
    """Build a deterministic prompt without exposing answer metadata."""
    stem, options = _display_text(question, language)
    qtype = str(question.get("qtype") or "")
    if language == "zh":
        instruction = (
            "请仅输出所有正确选项的字母，以逗号分隔。"
            if qtype == "multi_choice"
            else "请仅输出正确选项的字母。"
        )
        speech_label = "截至当前时刻可见的语音转写"
    else:
        instruction = (
            "Return only the keys of all correct options, separated by commas."
            if qtype == "multi_choice"
            else "Return only the key of the correct option."
        )
        speech_label = "Speech observed up to the current time"

    lines = [stem, "", "Options:" if language != "zh" else "选项："]
    lines.extend(f"{key}. {text}" for key, text in options)
    extra = question.get("extra") or {}
    if language == "zh":
        speech = extra.get("speech_context_original") or extra.get("speech_context")
    else:
        speech = extra.get("speech_context_en") or extra.get("speech_context")
    if speech:
        lines.extend(["", f"{speech_label}:", str(speech)])
    lines.extend(["", instruction])
    return "\n".join(lines)


def render_bbox_last_frame(
    question: dict[str, Any],
    release_dir: Path,
    *,
    bbox_norm: int = 1000,
) -> Image.Image:
    """Draw candidate boxes on the trigger frame for bbox-select questions."""
    if question.get("qtype") != "bbox_select":
        raise ValueError("render_bbox_last_frame requires qtype=bbox_select")
    paths = frame_paths(question, release_dir)
    image = Image.open(paths[-1]).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    sx = image.width / float(bbox_norm)
    sy = image.height / float(bbox_norm)
    line_width = max(2, round(min(image.width, image.height) / 250))
    for index, option in enumerate(question.get("options") or []):
        bbox = option.get("bbox") or {}
        if not bbox:
            continue
        coords = (
            float(bbox["x1"]) * sx,
            float(bbox["y1"]) * sy,
            float(bbox["x2"]) * sx,
            float(bbox["y2"]) * sy,
        )
        color = BBOX_COLORS[index % len(BBOX_COLORS)]
        draw.rectangle(coords, outline=color, width=line_width)
        draw.text((coords[0] + 4, coords[1] + 4), str(option.get("key")), fill=color, font=font)
    return image


def load_visual_inputs(
    question: dict[str, Any],
    release_dir: Path,
    *,
    bbox_norm: int = 1000,
) -> list[Image.Image]:
    """Load the full causal frame sequence, overlaying bbox choices on τ."""
    images = [Image.open(path).convert("RGB") for path in frame_paths(question, release_dir)]
    if question.get("qtype") == "bbox_select":
        images[-1] = render_bbox_last_frame(question, release_dir, bbox_norm=bbox_norm)
    return images
