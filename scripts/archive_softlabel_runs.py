from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path
from typing import Iterable, List, Set


def parse_paths(tsv_path: Path) -> List[Path]:
    paths: List[Path] = []
    for raw in tsv_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        # Expected: group, macro_f1, acc, path
        p = parts[-1].strip()
        if not p:
            continue
        paths.append(Path(p))
    return paths


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Archive softlabel run folders into one archive directory. "
            "Input is typically outputs/softlabels/_ensemble_bad_list.txt."
        )
    )
    ap.add_argument(
        "--list",
        type=Path,
        default=Path("outputs") / "softlabels" / "_ensemble_bad_list.txt",
        help="TSV list file (default: outputs/softlabels/_ensemble_bad_list.txt)",
    )
    ap.add_argument(
        "--archive-root",
        type=Path,
        default=Path("outputs") / "softlabels" / "_archive",
        help="Archive root folder (default: outputs/softlabels/_archive)",
    )
    ap.add_argument(
        "--mode",
        type=str,
        default="move",
        choices=["move", "copy"],
        help="Whether to move (default) or copy folders into the archive.",
    )
    ap.add_argument(
        "--tag",
        type=str,
        default="bad_list",
        help="Tag name used in archive folder name.",
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    list_path: Path = args.list
    if not list_path.exists():
        raise SystemExit(f"List file not found: {list_path}")

    src_paths = parse_paths(list_path)
    uniq: List[Path] = []
    seen: Set[str] = set()
    for p in src_paths:
        k = str(p).lower()
        if k in seen:
            continue
        seen.add(k)
        uniq.append(p)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    archive_dir = args.archive_root / f"{args.tag}_{stamp}"
    archive_dir.mkdir(parents=True, exist_ok=True)

    log_lines: List[str] = []
    log_lines.append(f"mode\t{args.mode}\n")
    log_lines.append(f"archive_dir\t{archive_dir.as_posix()}\n")
    log_lines.append(f"list\t{list_path.as_posix()}\n")

    # copy list into archive for provenance
    shutil.copy2(list_path, archive_dir / list_path.name)

    moved = 0
    skipped_missing = 0
    skipped_exists = 0

    for src in uniq:
        if not src.exists():
            skipped_missing += 1
            log_lines.append(f"missing\t{src.as_posix()}\n")
            continue

        dest = archive_dir / src.name
        if dest.exists():
            skipped_exists += 1
            log_lines.append(f"exists\t{src.as_posix()}\t->\t{dest.as_posix()}\n")
            continue

        if args.mode == "copy":
            shutil.copytree(src, dest)
            log_lines.append(f"copied\t{src.as_posix()}\t->\t{dest.as_posix()}\n")
        else:
            shutil.move(str(src), str(dest))
            log_lines.append(f"moved\t{src.as_posix()}\t->\t{dest.as_posix()}\n")
        moved += 1

    log_path = archive_dir / "archive_log.tsv"
    log_path.write_text("".join(log_lines), encoding="utf-8")

    print(f"ARCHIVE_DIR\t{archive_dir}")
    print(f"MODE\t{args.mode}")
    print(f"UNIQUE_PATHS\t{len(uniq)}")
    print(f"MOVED_OR_COPIED\t{moved}")
    print(f"SKIPPED_MISSING\t{skipped_missing}")
    print(f"SKIPPED_EXISTS\t{skipped_exists}")
    print(f"LOG\t{log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
