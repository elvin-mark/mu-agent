"""Resilient Unified Diff Parser & Multi-Stage Fuzzy Patching Engine for Mu Agent."""

import os
import re
from dataclasses import dataclass, field


@dataclass
class Hunk:
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: list[str] = field(default_factory=list)


@dataclass
class FilePatch:
    old_path: str
    new_path: str
    hunks: list[Hunk] = field(default_factory=list)


def parse_unified_diff(patch_text: str) -> list[FilePatch]:
    """Parse unified diff or Vibe-style patch text into structured FilePatch objects."""
    file_patches: list[FilePatch] = []
    current_patch: FilePatch | None = None
    current_hunk: Hunk | None = None

    lines = patch_text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        line_strip = line.strip()

        # Ignore wrapper markers
        if line_strip in ("*** Begin Patch", "*** End Patch", "```diff", "```"):
            i += 1
            continue

        # Check Vibe / LLM style file header: *** Update File: fib.py or *** File: fib.py
        if (
            line_strip.startswith("*** Update File:")
            or line_strip.startswith("*** File:")
            or line_strip.startswith("*** Create File:")
        ):
            file_name = line_strip.split(":", 1)[1].strip()
            current_patch = FilePatch(old_path=file_name, new_path=file_name)
            file_patches.append(current_patch)
            current_hunk = None
            i += 1
            continue

        # Check standard unified diff header: --- a/path and +++ b/path
        if line.startswith("--- "):
            old_path = line[4:].strip().removeprefix("a/")
            i += 1
            if i < len(lines) and lines[i].startswith("+++ "):
                new_path = lines[i][4:].strip().removeprefix("b/")
                current_patch = FilePatch(old_path=old_path, new_path=new_path)
                file_patches.append(current_patch)
                current_hunk = None
                i += 1
                continue

        # Check hunk header: @@ -old_start,old_count +new_start,new_count @@ or bare @@
        if line.startswith("@@"):
            match = re.match(r"^@@\s*-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@", line)
            if match:
                old_start = int(match.group(1))
                old_count = int(match.group(2)) if match.group(2) is not None else 1
                new_start = int(match.group(3))
                new_count = int(match.group(4)) if match.group(4) is not None else 1
            else:
                # Bare @@ without explicit line numbers
                old_start, old_count, new_start, new_count = 1, 1, 1, 1

            if current_patch is not None:
                current_hunk = Hunk(
                    old_start=old_start,
                    old_count=old_count,
                    new_start=new_start,
                    new_count=new_count,
                )
                current_patch.hunks.append(current_hunk)

        elif current_hunk is not None:
            if (
                line.startswith("+")
                or line.startswith("-")
                or line.startswith(" ")
                or line == ""
            ):
                # Context or diff line
                current_hunk.lines.append(line if line != "" else " ")

        i += 1

    return file_patches


def find_matching_index(
    file_lines: list[str], expected_lines: list[str], target_idx: int, window: int = 15
) -> int | None:
    """Multi-stage fuzzy matcher to locate expected lines in file_lines around target_idx."""
    n_file = len(file_lines)
    n_exp = len(expected_lines)
    if n_exp == 0:
        return max(0, min(target_idx, n_file))

    # Stage 1: Exact match at target_idx
    if 0 <= target_idx <= n_file - n_exp:
        if file_lines[target_idx : target_idx + n_exp] == expected_lines:
            return target_idx

    # Stage 2: Line-drift search window (±window)
    min_search = max(0, target_idx - window)
    max_search = min(n_file - n_exp, target_idx + window)

    for offset in range(window + 1):
        idx_f = target_idx + offset
        if (
            min_search <= idx_f <= max_search
            and file_lines[idx_f : idx_f + n_exp] == expected_lines
        ):
            return idx_f
        idx_b = target_idx - offset
        if (
            min_search <= idx_b <= max_search
            and file_lines[idx_b : idx_b + n_exp] == expected_lines
        ):
            return idx_b

    # Stage 3: Full file exact match search (for bare @@ or line drift > 15)
    for idx in range(n_file - n_exp + 1):
        if file_lines[idx : idx + n_exp] == expected_lines:
            return idx

    # Stage 4: Full file whitespace-insensitive match search
    norm_expected = [l.strip() for l in expected_lines]
    for idx in range(n_file - n_exp + 1):
        norm_file = [l.strip() for l in file_lines[idx : idx + n_exp]]
        if norm_file == norm_expected:
            return idx

    return None


def apply_hunk(file_lines: list[str], hunk: Hunk) -> tuple[list[str], bool, str]:
    """Apply a single hunk to file_lines using multi-stage fuzzy matching."""
    expected_old: list[str] = []
    replacement_new: list[str] = []

    for line in hunk.lines:
        prefix = line[0] if line else " "
        content = line[1:] if len(line) > 0 else ""
        if prefix in (" ", "-"):
            expected_old.append(content)
        if prefix in (" ", "+"):
            replacement_new.append(content)

    target_idx = max(0, hunk.old_start - 1)
    match_idx = find_matching_index(file_lines, expected_old, target_idx)

    if match_idx is None:
        return (
            file_lines,
            False,
            f"Failed to match hunk at line {hunk.old_start}. Expected context:\n"
            + "\n".join(expected_old[:5]),
        )

    # Apply replacement
    new_file_lines = (
        file_lines[:match_idx]
        + replacement_new
        + file_lines[match_idx + len(expected_old) :]
    )
    return new_file_lines, True, "Success"


def apply_patch_string(patch_text: str, root_dir: str = ".") -> tuple[bool, str]:
    """Apply a unified diff patch string across files in root_dir."""
    patches = parse_unified_diff(patch_text)
    if not patches:
        return False, "Error: No valid unified diff hunks found in patch string."

    applied_files = []
    for fp in patches:
        file_path = os.path.join(root_dir, fp.new_path)
        if not os.path.exists(file_path):
            file_path = os.path.join(root_dir, fp.old_path)
            if not os.path.exists(file_path):
                return False, f"Error: Target patch file '{fp.new_path}' not found."

        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            lines = content.splitlines()

            # Apply hunks
            success_count = 0
            for hunk in fp.hunks:
                lines, ok, err_msg = apply_hunk(lines, hunk)
                if not ok:
                    return (
                        False,
                        f"Error patching file '{fp.new_path}': {err_msg}",
                    )
                success_count += 1

            new_content = "\n".join(lines)
            if content.endswith("\n"):
                new_content += "\n"

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(new_content)

            applied_files.append(
                f"{fp.new_path} ({success_count} hunk{'s' if success_count > 1 else ''} applied)"
            )
        except Exception as e:
            return False, f"Error writing patch to '{fp.new_path}': {e!s}"

    return True, "Successfully applied patch to:\n" + "\n".join(applied_files)


def fuzzy_replace_string(
    content: str, target: str, replacement: str
) -> tuple[str, bool, str]:
    """Fuzzy replace target with replacement in content if exact match fails."""
    if target in content:
        return content.replace(target, replacement, 1), True, "Exact match"

    content_lines = content.splitlines()
    target_lines = target.splitlines()

    match_idx = find_matching_index(
        content_lines, target_lines, 0, window=len(content_lines)
    )
    if match_idx is not None:
        rep_lines = replacement.splitlines()
        new_lines = (
            content_lines[:match_idx]
            + rep_lines
            + content_lines[match_idx + len(target_lines) :]
        )
        res = "\n".join(new_lines)
        if content.endswith("\n"):
            res += "\n"
        return res, True, "Fuzzy match"

    return content, False, "Target content not found (exact or fuzzy)."
