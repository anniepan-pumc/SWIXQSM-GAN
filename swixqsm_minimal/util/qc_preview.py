"""Small self-contained visual QC pages; display scaling never changes data."""
import base64
import html
import struct
import zlib
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

from .data import load_volume, require_grid


def _png(array):
    array = np.ascontiguousarray(array, dtype=np.uint8)
    height, width, _ = array.shape

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff)

    rows = b"".join(b"\0" + array[index].tobytes() for index in range(height))
    content = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
               + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))
    return "data:image/png;base64," + base64.b64encode(content).decode()


def _display(array, axis, index, brain=None, contour=None):
    plane = np.take(array, index, axis=axis).T
    values = array[np.asarray(brain, dtype=bool)] if brain is not None and np.any(brain) else array.ravel()
    low, high = np.percentile(values, [1, 99])
    scaled = np.zeros_like(plane) if high <= low else np.clip((plane - low) / (high - low), 0, 1)
    gray = np.rint(scaled * 255).astype(np.uint8)
    rgb = np.stack([gray] * 3, axis=-1)
    if contour is not None:
        boundary = contour & ~ndimage.binary_erosion(contour)
        edge = np.take(boundary, index, axis=axis).T
        rgb[edge] = [255, 80, 40]
    return _png(np.flipud(rgb))


def write_preview(path, rows, cases, *, limit=12):
    """Show orthogonal array views for a bounded number of prepared pairs.

    These are independently windowed illustrations, not registration metrics,
    and are not radiological reorientations or substitutes for native-volume review.
    """
    if limit < 0:
        raise ValueError("preview limit must be nonnegative")
    parts = ["<!doctype html><meta charset='utf-8'><title>SWIXQSM Data QC</title>",
             "<style>body{font:15px system-ui;margin:24px;background:#fafafa}table{border-collapse:collapse}td,th{padding:8px;vertical-align:top}img{width:220px;max-height:260px;object-fit:contain;background:#000}section{margin:24px 0;padding:16px;background:white;border:1px solid #ddd}code{word-break:break-all}</style>",
             "<h1>配准与数据质量预览</h1><p>以下为当前数组网格的正交切面，各图仅为显示独立取1/99百分位。显示缩放不进入训练或数值评价。橙色轮廓仅在提供独立QSM脑mask时显示。",
             "这些示意图不能自动证明配准成功；请在三维原图中核对脑室、基底节、小病灶及参考ROI。当前网格角度不等于原始头位或B₀角度。</p>"]
    for row, case in list(zip(rows, cases))[:limit]:
        label = html.escape(str(row.get("case_id") or row.get("subject_id")))
        parts.append(f"<section><h2>{label}</h2><p>QC记录 <code>{html.escape(str(case.get('key', '')))}</code></p>")
        try:
            ref, swi = load_volume(row["swi_path"])
            _, brain = load_volume(row["swi_mask_path"])
            target_path = row.get("registration_image_path") or row["qsm_path"]
            image, target = load_volume(target_path)
            require_grid(ref, image)
            contour = None
            if row.get("qsm_brain_mask_path"):
                mask_ref, value = load_volume(row["qsm_brain_mask_path"])
                require_grid(ref, mask_ref)
                contour = value > 0
            parts.append("<table><tr><th>当前数组切面</th><th>SWI（可叠加独立脑mask轮廓）</th><th>" +
                         ("配准后辅助解剖图" if row.get("registration_image_path") else "Reference QSM（对比度不同）") + "</th></tr>")
            for axis in (2, 1, 0):
                index = swi.shape[axis] // 2
                left = _display(swi, axis, index, brain > 0, contour)
                right = _display(target, axis, index, brain > 0)
                parts.append(f"<tr><td>axis {axis}, index {index}</td><td><img src='{left}'></td><td><img src='{right}'></td></tr>")
            parts.append("</table>")
        except (ValueError, TypeError, OSError, EOFError, KeyError, IndexError,
                nib.filebasedimages.ImageFileError) as exc:
            parts.append("<p>无法生成可比预览 <code>" + html.escape(str(exc)) + "</code></p>")
        parts.append("</section>")
    parts.append(f"<p>展示上限为{limit}例；完整逐例结果见report.json。预览只是有界示例，未展示的病例同样需要完成相应审核。</p>")
    Path(path).write_text("\n".join(parts), encoding="utf-8")
