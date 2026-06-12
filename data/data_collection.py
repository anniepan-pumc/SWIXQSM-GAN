"""
从 BIDS 目录 csvd-bids 中提取每个被试的 QSM、Chimap、SWI、SWI_brain_mask、QSM_brain_mask，
复制到目标目录并重命名为统一文件名。
"""

import argparse
import shutil
from pathlib import Path

# 源数据根目录
DEFAULT_SOURCE_DIR = "/Volumes/MOVESPEED/QSM_multicohort_data/csvd-bids"

# 目标目录下每个被试内使用的统一文件名
OUT_NAMES = {
    "qsm_nii": "QSM.nii.gz",
    "chimap_nii": "Chimap.nii.gz",
    "swi_nii": "SWI.nii.gz",
    "swi_brain_mask_nii": "SWI_brain_mask.nii.gz",
    "qsm_brain_mask_nii": "QSM_brain_mask.nii.gz",
}


def _first_existing(*paths):
    """返回第一个存在的 Path，若都不存在则返回 None。"""
    for p in paths:
        if p is not None and Path(p).exists():
            return Path(p)
    return None


def _glob_first(parent: Path, pattern: str):
    """在 parent 下按 pattern 查找，返回第一个匹配的 Path，否则 None。"""
    if not parent.exists():
        return None
    matches = list(parent.glob(pattern))
    return Path(matches[0]) if matches else None


def get_subject_paths(sub_dir: Path, subject_id: str) -> dict:
    """
    获取单个被试在 BIDS 源目录中的文件路径（仅 nii.gz）。
    subject_id 格式为 sub-xxxx。
    返回 dict：key 为 out_NAMES 的 key，value 为源文件 Path 或 None。
    """
    qsm_dir = sub_dir / "QSM"
    result_dir = qsm_dir / "result"
    swi_dir = sub_dir / "SWI"

    # Chimap: 支持两种命名
    # 1) sub-xxx+LaplacianMEDI+VSHARPSTIsuite+Star-QSM_Chimap.nii.gz
    # 2) sub-xxx+Laplacian (MEDI)+VSHARP (STI suite)+Star-QSM_Chimap.nii.gz
    chimap_nii = _glob_first(qsm_dir, f"{subject_id}+*Chimap.nii.gz")
    if chimap_nii is None:
        chimap_nii = _glob_first(qsm_dir, "*Chimap.nii.gz")

    # QSM brain_mask: 优先 QSM/brain_mask.nii.gz
    qsm_brain_mask_nii = _first_existing(
        qsm_dir / "brain_mask.nii.gz",
        result_dir / "QSM_mask.nii.gz",
    )

    return {
        "qsm_nii": _first_existing(result_dir / "QSM.nii.gz"),
        "chimap_nii": chimap_nii,
        "swi_nii": _first_existing(swi_dir / f"{subject_id}_SWI.nii.gz"),
        "swi_brain_mask_nii": _first_existing(
            swi_dir / f"{subject_id}_SWI_brain_mask.nii.gz"
        ),
        "qsm_brain_mask_nii": qsm_brain_mask_nii,
    }


def extract_and_rename(
    source_root: Path,
    out_root: Path,
    subject_ids: list[str] | None = None,
    copy: bool = True,
) -> list[dict]:
    """
    从 source_root（csvd-bids）中提取每个 sub-xxxx 的 QSM/Chimap/SWI/mask，复制到 out_root 并重命名。

    :param source_root: BIDS 根目录，例如 /Volumes/.../csvd-bids
    :param out_root: 输出根目录；每个被试为 out_root/sub-xxxx/，其下为统一文件名
    :param subject_ids: 若指定则只处理这些 subject_id；否则扫描 source_root 下所有 sub-*
    :param copy: True 复制文件，False 仅收集路径不写入
    :return: 列表，每项为 {"subject_id": str, "out_dir": Path, "paths": get_subject_paths 的返回值, "copied": {key: bool}}
    """
    source_root = Path(source_root)
    out_root = Path(out_root)

    if subject_ids is None:
        subject_ids = sorted(
            d.name for d in source_root.iterdir() if d.is_dir() and d.name.startswith("sub-")
        )

    results = []
    for sid in subject_ids:
        sub_dir = source_root / sid
        if not sub_dir.is_dir():
            results.append(
                {
                    "subject_id": sid,
                    "out_dir": None,
                    "paths": {},
                    "copied": {},
                    "error": f"not a directory: {sub_dir}",
                }
            )
            continue

        paths = get_subject_paths(sub_dir, sid)
        out_dir = out_root / sid
        copied = {k: False for k in OUT_NAMES}

        if copy:
            out_dir.mkdir(parents=True, exist_ok=True)
            for key, src in paths.items():
                if src is not None and src.exists():
                    dst = out_dir / OUT_NAMES[key]
                    try:
                        shutil.copy2(src, dst)
                        copied[key] = True
                    except Exception as e:
                        copied[key] = False
                        results.append(
                            {
                                "subject_id": sid,
                                "out_dir": out_dir,
                                "paths": paths,
                                "copied": copied,
                                "error": str(e),
                            }
                        )
                        break
            else:
                results.append(
                    {
                        "subject_id": sid,
                        "out_dir": out_dir,
                        "paths": paths,
                        "copied": copied,
                        "error": None,
                    }
                )
        else:
            results.append(
                {
                    "subject_id": sid,
                    "out_dir": out_dir,
                    "paths": paths,
                    "copied": copied,
                    "error": None,
                }
            )

    return results


def main():
    parser = argparse.ArgumentParser(
        description="从 csvd-bids 中提取每个被试的 QSM、Chimap、SWI、SWI_brain_mask、QSM_brain_mask 并重命名复制到目标目录"
    )
    parser.add_argument(
        "--source",
        type=str,
        default=DEFAULT_SOURCE_DIR,
        help="BIDS 根目录（默认: %(default)s）",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="输出根目录；默认在 source 同目录下 csvd-bids-extracted",
    )
    parser.add_argument(
        "--subjects",
        type=str,
        nargs="*",
        default=None,
        help="只处理指定的 subject_id 列表，如 sub-001005 sub-001012",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只扫描并打印将要复制的文件，不实际复制",
    )
    args = parser.parse_args()

    source = Path(args.source)
    out_root = Path(args.out) if args.out else source.parent/Path(str(source.stem)+'_extracted')

    if not source.exists():
        print(f"错误: 源目录不存在: {source}")
        return 1

    print(f"源目录: {source}")
    print(f"输出目录: {out_root}")
    print(f"干跑(不复制): {args.dry_run}")
    if args.subjects:
        print(f"指定被试: {args.subjects}")

    results = extract_and_rename(
        source_root=source,
        out_root=out_root,
        subject_ids=args.subjects,
        copy=not args.dry_run,
    )

    # 统计与简要报告
    ok = sum(1 for r in results if r.get("error") is None)
    missing = []
    for r in results:
        if r.get("error"):
            print(f"  [跳过] {r['subject_id']}: {r['error']}")
            continue
        p = r.get("paths") or {}
        c = r.get("copied") or {}
        for key in OUT_NAMES:
            if p.get(key) is None:
                missing.append((r["subject_id"], key))
            elif args.dry_run and c.get(key):
                print(f"  [会复制] {r['subject_id']} {key} -> {OUT_NAMES[key]}")

    print(f"\n处理被试数: {len(results)}, 成功: {ok}")
    if missing:
        print(f"缺失文件数: {len(missing)}")
        for sid, key in missing[:20]:
            print(f"  缺失: {sid} {key}")
        if len(missing) > 20:
            print(f"  ... 及其他 {len(missing) - 20} 项")
    return 0


if __name__ == "__main__":
    exit(main())
