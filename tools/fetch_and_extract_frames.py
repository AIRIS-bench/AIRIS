#!/usr/bin/env python3
"""Download referenced YouTube videos and reconstruct AIRIS benchmark frames.

The repository does not redistribute videos or frames. Users are responsible for
complying with YouTube's terms and all applicable rights and laws.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import secrets
import stat
import subprocess
import sys
import threading
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Sequence

VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
VIDEO_EXTENSIONS = {".mp4", ".webm", ".mkv", ".mov", ".m4v"}
_PRINT_LOCK = threading.Lock()


@dataclass(frozen=True)
class FrameSpec:
    frame_path: str
    clip_id: str
    clip_time_ms: int
    video_id: str
    url: str
    source_time_ms: int
    width: int
    height: int


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


def resolve_from_root(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def report_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return "<external>"


def load_frame_specs(path: Path) -> list[FrameSpec]:
    specs: list[FrameSpec] = []
    seen: set[str] = set()
    for row in read_jsonl(path):
        spec = FrameSpec(
            frame_path=str(row["frame_path"]),
            clip_id=str(row["clip_id"]),
            clip_time_ms=int(row["clip_time_ms"]),
            video_id=str(row["source_video_id"]),
            url=str(row["youtube_url"]),
            source_time_ms=int(row["source_time_ms"]),
            width=int(row["width"]),
            height=int(row["height"]),
        )
        if spec.frame_path in seen:
            raise ValueError(f"duplicate frame_path: {spec.frame_path}")
        if not VIDEO_ID_RE.fullmatch(spec.video_id):
            raise ValueError(f"invalid YouTube video ID: {spec.video_id!r}")
        expected_url = f"https://www.youtube.com/watch?v={spec.video_id}"
        if spec.url != expected_url:
            raise ValueError(f"non-canonical URL for {spec.video_id}: {spec.url}")
        expected_path = f"frames/{spec.clip_id}/{spec.clip_id}__{spec.clip_time_ms:09d}.jpg"
        if spec.frame_path != expected_path:
            raise ValueError(f"invalid frame path: {spec.frame_path!r}; expected {expected_path!r}")
        if spec.source_time_ms < 0 or spec.width <= 0 or spec.height <= 0:
            raise ValueError(f"invalid frame metadata: {spec.frame_path}")
        seen.add(spec.frame_path)
        specs.append(spec)
    return specs


def safe_frame_output(root: Path, relative: str) -> Path:
    rel = PurePosixPath(relative)
    if rel.is_absolute() or ".." in rel.parts or len(rel.parts) != 3 or rel.parts[0] != "frames":
        raise ValueError(f"unsafe frame path: {relative!r}")
    target = root.joinpath(*rel.parts)
    current = root
    for part in rel.parts[:-1]:
        current = current / part
        if current.exists() and current.is_symlink():
            raise ValueError(f"symlink is not allowed in frame output: {relative!r}")
    return target


def validate_sources(sources: list[dict[str, Any]], specs: list[FrameSpec]) -> None:
    source_map: dict[str, dict[str, Any]] = {}
    dimensions: dict[str, set[tuple[int, int]]] = defaultdict(set)
    for spec in specs:
        dimensions[spec.video_id].add((spec.width, spec.height))
    for row in sources:
        video_id = str(row.get("video_id", ""))
        if video_id in source_map:
            raise ValueError(f"duplicate source video ID: {video_id}")
        if not VIDEO_ID_RE.fullmatch(video_id):
            raise ValueError(f"invalid source video ID: {video_id!r}")
        expected_url = f"https://www.youtube.com/watch?v={video_id}"
        if row.get("platform") != "youtube" or row.get("url") != expected_url:
            raise ValueError(f"invalid source platform/URL for {video_id}")
        width, height = int(row.get("expected_width", 0)), int(row.get("expected_height", 0))
        if width <= 0 or height <= 0 or dimensions.get(video_id) != {(width, height)}:
            raise ValueError(f"source/frame dimensions disagree for {video_id}")
        clip_ids = row.get("clip_ids") or []
        if len(clip_ids) != len(set(map(str, clip_ids))) or int(row.get("n_clips", -1)) != len(clip_ids):
            raise ValueError(f"invalid clip list for {video_id}")
        source_map[video_id] = row
    if set(source_map) != set(dimensions):
        raise ValueError("source_videos.jsonl and frame_index.jsonl contain different video IDs")


def find_video(sources_dir: Path, video_id: str) -> Path | None:
    candidates = sorted(
        path
        for path in sources_dir.glob(f"{video_id}.*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )
    if len(candidates) > 1:
        raise RuntimeError(f"multiple source files for {video_id}: {candidates}")
    return candidates[0] if candidates else None


def print_status(message: str) -> None:
    with _PRINT_LOCK:
        print(message, flush=True)


def download_one(
    row: dict[str, Any],
    sources_dir: Path,
    cookies: Path | None,
    cookies_from_browser: str | None,
    overwrite: bool,
) -> dict[str, Any]:
    video_id = str(row["video_id"])
    url = str(row["url"])
    existing = find_video(sources_dir, video_id)
    if existing is not None and not overwrite:
        return {"video_id": video_id, "status": "existing", "file": existing.name}

    width = int(row["expected_width"])
    height = int(row["expected_height"])
    exact = f"bestvideo[width={width}][height={height}]/best[width={width}][height={height}]"
    format_selector = f"{exact}/bestvideo/best"
    command = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--no-playlist",
        "--continue",
        "--write-info-json",
        "--format",
        format_selector,
        "--output",
        str(sources_dir / f"{video_id}.%(ext)s"),
    ]
    command.append("--force-overwrites" if overwrite else "--no-overwrites")
    if cookies is not None:
        command.extend(["--cookies", str(cookies)])
    if cookies_from_browser:
        command.extend(["--cookies-from-browser", cookies_from_browser])
    command.extend(["--", url])

    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode != 0:
        tail = (result.stderr or result.stdout).strip().splitlines()[-3:]
        return {"video_id": video_id, "status": "failed", "error": " | ".join(tail)}
    downloaded = find_video(sources_dir, video_id)
    if downloaded is None:
        return {"video_id": video_id, "status": "failed", "error": "yt-dlp produced no video file"}
    return {"video_id": video_id, "status": "downloaded", "file": downloaded.name}


def run_download(args: argparse.Namespace, sources: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        import yt_dlp  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("yt-dlp is missing; run: pip install -r requirements.txt") from exc

    args.sources_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = {
            executor.submit(
                download_one,
                row,
                args.sources_dir,
                args.cookies,
                args.cookies_from_browser,
                args.overwrite,
            ): str(row["video_id"])
            for row in sources
        }
        for future in concurrent.futures.as_completed(futures):
            video_id = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # keep other downloads running
                result = {"video_id": video_id, "status": "failed", "error": str(exc)}
            results.append(result)
            print_status(f"[{result['status']}] {video_id}")
    results.sort(key=lambda row: str(row["video_id"]))
    failed = [row for row in results if row["status"] == "failed"]
    return {"n_sources": len(sources), "n_failed": len(failed), "items": results}


def save_decoded_frame(
    frame: Any,
    frame_time_ms: int,
    spec: FrameSpec,
    root: Path,
    overwrite: bool,
    resize_to_metadata: bool,
    jpeg_quality: int,
    max_time_error_ms: int,
) -> str:
    target = safe_frame_output(root, spec.frame_path)
    error_ms = abs(frame_time_ms - spec.source_time_ms)
    if error_ms > max_time_error_ms:
        raise RuntimeError(
            f"{spec.frame_path}: nearest decoded frame differs by {error_ms} ms "
            f"(limit {max_time_error_ms} ms)"
        )
    image = frame.to_image().convert("RGB")
    expected_size = (spec.width, spec.height)
    if image.size != expected_size:
        if not resize_to_metadata:
            raise RuntimeError(
                f"{spec.video_id}: decoded size {image.size} != expected {expected_size}; "
                "select the matching YouTube format or pass --resize-to-metadata"
            )
        from PIL import Image

        image = image.resize(expected_size, Image.Resampling.LANCZOS)

    frames_root = root / "frames"
    frames_root.mkdir(exist_ok=True)
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    frames_fd = os.open(frames_root, directory_flags)
    clip_fd: int | None = None
    temporary: str | None = None
    try:
        try:
            os.mkdir(spec.clip_id, mode=0o755, dir_fd=frames_fd)
        except FileExistsError:
            pass
        clip_fd = os.open(spec.clip_id, directory_flags, dir_fd=frames_fd)
        try:
            current = os.stat(target.name, dir_fd=clip_fd, follow_symlinks=False)
        except FileNotFoundError:
            current = None
        if current is not None and not stat.S_ISREG(current.st_mode):
            raise RuntimeError(f"refusing to replace non-regular output: {spec.frame_path}")
        if current is not None and not overwrite:
            return "existing"

        temporary = f".{target.name}.{secrets.token_hex(12)}.tmp"
        file_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        output_fd = os.open(temporary, file_flags, 0o644, dir_fd=clip_fd)
        with os.fdopen(output_fd, "wb") as handle:
            image.save(handle, format="JPEG", quality=jpeg_quality, subsampling=0, optimize=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target.name, src_dir_fd=clip_fd, dst_dir_fd=clip_fd)
        temporary = None
    finally:
        if temporary is not None and clip_fd is not None:
            try:
                os.unlink(temporary, dir_fd=clip_fd)
            except FileNotFoundError:
                pass
        if clip_fd is not None:
            os.close(clip_fd)
        os.close(frames_fd)
    return "written"


def frame_output_is_regular(root: Path, spec: FrameSpec) -> bool:
    target = safe_frame_output(root, spec.frame_path)
    try:
        return stat.S_ISREG(target.lstat().st_mode)
    except FileNotFoundError:
        return False


def extract_video_frames(
    video_id: str,
    specs: list[FrameSpec],
    args: argparse.Namespace,
) -> dict[str, Any]:
    try:
        import av
    except ImportError as exc:
        raise RuntimeError("PyAV is missing; run: pip install -r requirements.txt") from exc

    pending = [
        spec
        for spec in sorted(specs, key=lambda item: (item.source_time_ms, item.frame_path))
        if args.overwrite or not frame_output_is_regular(args.root, spec)
    ]
    if not pending:
        return {"video_id": video_id, "status": "existing", "n_frames": len(specs)}
    source = find_video(args.sources_dir, video_id)
    if source is None:
        return {"video_id": video_id, "status": "failed", "error": "source video is missing"}

    written = 0
    with av.open(str(source)) as container:
        stream = next((item for item in container.streams if item.type == "video"), None)
        if stream is None:
            return {"video_id": video_id, "status": "failed", "error": "no video stream"}
        stream_time_base = float(stream.time_base)
        stream_start_pts = int(stream.start_time or 0)
        seek_seconds = max(0.0, pending[0].source_time_ms / 1000.0 - 2.0)
        if seek_seconds:
            seek_pts = stream_start_pts + int(seek_seconds / stream_time_base)
            container.seek(seek_pts, stream=stream, backward=True, any_frame=False)

        index = 0
        previous = None
        previous_ms: int | None = None
        for frame in container.decode(stream):
            if frame.pts is None or frame.time_base is None:
                continue
            current_ms = int(round(float((frame.pts - stream_start_pts) * frame.time_base) * 1000))
            while index < len(pending) and pending[index].source_time_ms <= current_ms:
                spec = pending[index]
                chosen, chosen_ms = frame, current_ms
                if previous is not None and previous_ms is not None:
                    if abs(previous_ms - spec.source_time_ms) <= abs(current_ms - spec.source_time_ms):
                        chosen, chosen_ms = previous, previous_ms
                status = save_decoded_frame(
                    chosen,
                    chosen_ms,
                    spec,
                    args.root,
                    args.overwrite,
                    args.resize_to_metadata,
                    args.jpeg_quality,
                    args.max_time_error_ms,
                )
                written += int(status == "written")
                index += 1
            if index >= len(pending):
                break
            previous, previous_ms = frame, current_ms

        while index < len(pending) and previous is not None and previous_ms is not None:
            spec = pending[index]
            status = save_decoded_frame(
                previous,
                previous_ms,
                spec,
                args.root,
                args.overwrite,
                args.resize_to_metadata,
                args.jpeg_quality,
                args.max_time_error_ms,
            )
            written += int(status == "written")
            index += 1
        if index != len(pending):
            return {
                "video_id": video_id,
                "status": "failed",
                "error": f"decoded {index}/{len(pending)} pending frames",
            }
    return {"video_id": video_id, "status": "extracted", "n_written": written, "n_frames": len(specs)}


def run_extract(args: argparse.Namespace, specs: list[FrameSpec]) -> dict[str, Any]:
    grouped: dict[str, list[FrameSpec]] = defaultdict(list)
    for spec in specs:
        grouped[spec.video_id].append(spec)
    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = {
            executor.submit(extract_video_frames, video_id, rows, args): video_id
            for video_id, rows in grouped.items()
        }
        for future in concurrent.futures.as_completed(futures):
            video_id = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # keep other videos running
                result = {"video_id": video_id, "status": "failed", "error": str(exc)}
            results.append(result)
            print_status(f"[{result['status']}] {video_id}")
    results.sort(key=lambda row: str(row["video_id"]))
    failed = [row for row in results if row["status"] == "failed"]
    return {"n_sources": len(grouped), "n_frames": len(specs), "n_failed": len(failed), "items": results}


def run_verify(args: argparse.Namespace, specs: list[FrameSpec]) -> dict[str, Any]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is missing; run: pip install -r requirements.txt") from exc

    failures: list[dict[str, str]] = []
    for spec in specs:
        try:
            path = safe_frame_output(args.root, spec.frame_path)
            if not path.is_file() or path.is_symlink():
                raise FileNotFoundError("missing or non-regular frame")
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                if image.format != "JPEG":
                    raise ValueError(f"expected JPEG, got {image.format}")
                if image.size != (spec.width, spec.height):
                    raise ValueError(f"size {image.size} != {(spec.width, spec.height)}")
        except Exception as exc:
            failures.append({"frame_path": spec.frame_path, "error": str(exc)})
    return {
        "n_frames": len(specs),
        "n_verified": len(specs) - len(failures),
        "n_failed": len(failures),
        "failures": failures[:100],
        "failures_truncated": max(0, len(failures) - 100),
    }


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("plan", "download", "extract", "verify", "all"), default="plan")
    parser.add_argument("--root", type=Path, default=default_root, help="release root")
    parser.add_argument("--frame-index", default="data/frame_index.jsonl")
    parser.add_argument("--source-videos", default="data/source_videos.jsonl")
    parser.add_argument("--sources-dir", default=".cache/youtube")
    parser.add_argument("--report", default="reports/media_build.json")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--cookies", type=Path, help="optional Netscape cookies file; never committed")
    parser.add_argument("--cookies-from-browser", help="optional yt-dlp browser name")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resize-to-metadata", action="store_true")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--max-time-error-ms", type=int, default=100)
    parser.add_argument("--allow-unavailable", action="store_true")
    args = parser.parse_args(argv)
    args.root = args.root.resolve()
    args.frame_index = resolve_from_root(args.root, args.frame_index)
    args.source_videos = resolve_from_root(args.root, args.source_videos)
    args.sources_dir = resolve_from_root(args.root, args.sources_dir)
    args.report = resolve_from_root(args.root, args.report)
    if args.jobs < 1:
        parser.error("--jobs must be at least 1")
    if not 1 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-quality must be in [1, 100]")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    specs = load_frame_specs(args.frame_index)
    sources = read_jsonl(args.source_videos)
    validate_sources(sources, specs)

    report: dict[str, Any] = {
        "stage": args.stage,
        "n_sources": len(sources),
        "n_frames": len(specs),
        "notice": "YouTube media is not distributed by this repository.",
    }
    if args.stage == "plan":
        report["plan"] = {
            "download_dir": report_path(args.sources_dir, args.root),
            "frames_dir": "frames",
            "source_time_rule": "source_time_ms from data/frame_index.jsonl",
        }
        write_report(args.report, report)
        print(f"plan: {len(sources)} YouTube videos, {len(specs)} unique frames")
        return 0

    download_failed = 0
    hard_failed = 0
    if args.stage in {"download", "all"}:
        report["download"] = run_download(args, sources)
        download_failed = int(report["download"]["n_failed"])
        if download_failed and args.stage == "all":
            write_report(args.report, report)
            print("download failed; extraction was not started", file=sys.stderr)
            return 1
    if args.stage in {"extract", "all"}:
        report["extract"] = run_extract(args, specs)
        hard_failed += int(report["extract"]["n_failed"])
    if args.stage in {"verify", "all"}:
        report["verify"] = run_verify(args, specs)
        hard_failed += int(report["verify"]["n_failed"])

    write_report(args.report, report)
    failed = hard_failed + (0 if args.stage == "download" and args.allow_unavailable else download_failed)
    if failed:
        print(f"completed with {failed} failure(s); see {args.report}", file=sys.stderr)
        return 1
    print(f"completed; report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
