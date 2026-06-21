#!/usr/bin/env python3
import argparse
import json
import re
from difflib import get_close_matches
from pathlib import Path
from typing import Dict, Any, Tuple, List

import numpy as np
from PIL import Image


def _normalize_name(name: str) -> str:
    # Lowercase
    s = name.lower()
    # Remove common numeric/id suffixes/prefixes
    s = re.sub(r"[._-]?\d{1,4}$", "", s)            # trailing digits like _001, .12, -3
    s = re.sub(r"\(\d{1,4}\)$", "", s)            # trailing (001)
    s = re.sub(r"\b(id|inst|instance)[ _-]*\d+\b", "", s)
    # Remove path-like segments
    s = s.split('/')[-1]
    # Replace non-alnum with space
    s = re.sub(r"[^a-z0-9]+", " ", s)
    # Collapse spaces
    s = re.sub(r"\s+", " ", s).strip()
    # Drop leading 'sm' designator if present
    if s.startswith('sm '):
        s = s[3:]
    return s


def load_sm_to_taxonomy(mapping_path: Path) -> Tuple[Dict[str, str], Dict[str, str], List[str]]:
    with open(mapping_path, 'r') as f:
        data = json.load(f)
    sm_to_taxonomy: Dict[str, str] = {}
    norm_to_taxonomy: Dict[str, str] = {}
    for taxonomy_name, entries in data.items():
        for entry in entries:
            sm_name = entry.get('object name', '')
            if not sm_name:
                continue
            sm_to_taxonomy[sm_name] = taxonomy_name
            norm = _normalize_name(sm_name)
            # Only set if not present to keep the first encountered mapping for a normalized key
            if norm and norm not in norm_to_taxonomy:
                norm_to_taxonomy[norm] = taxonomy_name
    return sm_to_taxonomy, norm_to_taxonomy, list(norm_to_taxonomy.keys())


def analyze_view_dir(scene_dir: Path, sm_to_taxonomy: Dict[str, str], norm_to_taxonomy: Dict[str, str], norm_keys: List[str],
                     min_w: int, min_h: int, min_area_ratio: float, min_ar: float) -> Dict[str, Any]:
    seg_path = scene_dir / 'seg.png'
    seen_path = scene_dir / 'seenable_obj_dict.json'
    out: Dict[str, Any] = {}
    if not seg_path.exists():
        return {'__summary__': {'view': scene_dir.name, 'error': 'missing_seg_png'}}
    if not seen_path.exists():
        return {'__summary__': {'view': scene_dir.name, 'error': 'missing_seenable_obj_dict'}}

    seg = np.array(Image.open(seg_path).convert('RGB'))
    H, W, _ = seg.shape
    img_area = H * W

    try:
        seen = json.loads(seen_path.read_text())
    except Exception:
        return {'__summary__': {'view': scene_dir.name, 'error': 'invalid_seenable_obj_dict'}}

    structural = {
        'floor', 'wall', 'ceiling', 'pillar', 'column', 'beam', 'stair',
        'stairs', 'railing', 'fence', 'roof'
    }

    kept = 0
    total = 0
    for sm, rgb in seen.items():
        total += 1
        color = tuple(rgb)
        mask = (seg[:, :, 0] == color[0]) & (seg[:, :, 1] == color[1]) & (seg[:, :, 2] == color[2])
        px = int(mask.sum())
        entry: Dict[str, Any] = {
            'color': color,
            'pixels': px,
            'taxonomy': None,
        }

        # Resolve taxonomy by exact match, normalized match, or fuzzy match
        tax = sm_to_taxonomy.get(sm)
        taxonomy_resolution = None
        if tax is None:
            sm_norm = _normalize_name(sm)
            tax = norm_to_taxonomy.get(sm_norm)
            if tax is not None:
                taxonomy_resolution = {'method': 'normalized_exact', 'normalized_key': sm_norm}
            else:
                # Containment heuristic (handles concatenations like "lamptable" -> "table")
                # Prefer the longest contained key (most specific); tie-breaker: rightmost occurrence
                contained: list[tuple[int, str]] = []
                for k in norm_keys:
                    # Skip very short keys and keys that are just the 'sm' token
                    if len(k) < 5 or k.startswith('sm '):
                        continue
                    idx = sm_norm.find(k) if sm_norm else -1
                    if idx >= 0:
                        contained.append((idx, k))
                if contained:
                    # Choose candidate with maximum length first; tie-breaker: largest start index (rightmost)
                    contained.sort(key=lambda t: (len(t[1]), t[0]), reverse=True)
                    mk = contained[0][1]
                    tax = norm_to_taxonomy.get(mk)
                    taxonomy_resolution = {'method': 'normalized_contained', 'normalized_key': sm_norm, 'matched_key': mk}
                if tax is None:
                    # Fuzzy over normalized keys
                    matches = get_close_matches(sm_norm, norm_keys, n=1, cutoff=0.85) if sm_norm else []
                    if matches:
                        mk = matches[0]
                        tax = norm_to_taxonomy.get(mk)
                        taxonomy_resolution = {'method': 'normalized_fuzzy', 'normalized_key': sm_norm, 'matched_key': mk}
        else:
            taxonomy_resolution = {'method': 'exact'}

        entry['taxonomy'] = tax
        if taxonomy_resolution is not None:
            entry['taxonomy_resolution'] = taxonomy_resolution
        if px == 0:
            entry['status'] = 'filtered'
            entry['reason'] = 'no_segmentation_pixels'
            out[sm] = entry
            continue

        ys, xs = np.where(mask)
        miny, maxy = int(ys.min()), int(ys.max())
        minx, maxx = int(xs.min()), int(xs.max())
        w = maxx - minx + 1
        h = maxy - miny + 1
        area = w * h
        ar = (min(w, h) / max(w, h)) if max(w, h) > 0 else 0.0
        reasons = []
        if w < min_w:
            reasons.append('too_narrow_width')
        if h < min_h:
            reasons.append('too_short_height')
        if area < img_area * min_area_ratio:
            reasons.append('too_small_area')
        if ar < min_ar:
            reasons.append('too_low_aspect_ratio')

        bbox = {
            'bbox': [minx, miny, maxx, maxy],
            'w': w,
            'h': h,
            'area': area,
            'aspect_ratio': ar,
        }

        if reasons:
            entry['status'] = 'filtered'
            entry['reason'] = ';'.join(reasons)
            entry['bbox'] = bbox
        else:
            entry['status'] = 'kept'
            entry['chosen_bbox'] = bbox
            kept += 1

        tax = entry.get('taxonomy')
        if tax:
            tl = tax.lower()
            entry['is_structural'] = any(s in tl for s in structural)
        out[sm] = entry

    out['__summary__'] = {
        'view': scene_dir.name,
        'usable_objects': kept,
        'total_objects_considered': total,
        'usable_rate': round((kept / total * 100) if total else 0.0, 2),
    }
    return out


def main():
    parser = argparse.ArgumentParser(description='Sim-image per-view filter diagnostics')
    parser.add_argument('--input_root', required=True, help='Root dir with scene categories and views (e.g., sim_images)')
    parser.add_argument('--output_dir', required=True, help='Directory to write diagnostics JSONs and index TSV')
    parser.add_argument('--mapping_path', required=False,
                        default=str(Path(__file__).resolve().parents[1] / 'modules' / 'sim_scene_object' / 'data' / 'new_all_object_list.json'))
    parser.add_argument('--min_width', type=int, default=40)
    parser.add_argument('--min_height', type=int, default=40)
    parser.add_argument('--min_area_ratio', type=float, default=0.005)
    parser.add_argument('--min_aspect_ratio', type=float, default=0.2)
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sm_to_taxonomy, norm_to_taxonomy, norm_keys = load_sm_to_taxonomy(Path(args.mapping_path))

    index_rows = []
    for scene_cat_dir in sorted([p for p in input_root.iterdir() if p.is_dir()]):
        for view_dir in sorted([p for p in scene_cat_dir.iterdir() if p.is_dir()]):
            diag = analyze_view_dir(
                view_dir, sm_to_taxonomy, norm_to_taxonomy, norm_keys,
                args.min_width, args.min_height, args.min_area_ratio, args.min_aspect_ratio,
            )
            view_name = f'{scene_cat_dir.name}_{view_dir.name}'
            out_path = output_dir / f'{view_name}.json'
            out_path.write_text(json.dumps(diag, indent=2))
            s = diag.get('__summary__', {})
            index_rows.append((view_name, s.get('usable_objects', 0), s.get('total_objects_considered', 0), s.get('usable_rate', 0)))

    tsv_path = output_dir / 'image_level_usable_counts.tsv'
    with open(tsv_path, 'w') as w:
        w.write('image_view\tusable_objects\ttotal_objects\tusable_rate_percent\n')
        for v, k, t, r in index_rows:
            w.write(f'{v}\t{k}\t{t}\t{r}\n')

    print(str(tsv_path))


if __name__ == '__main__':
    main()

