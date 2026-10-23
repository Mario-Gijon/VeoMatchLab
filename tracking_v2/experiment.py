"""Run a small, local team-discovery experiment on saved person crops."""
from __future__ import annotations

import argparse
import csv
import html
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from .team_discovery import (
    TEAM_A,
    TEAM_B,
    UNKNOWN,
    CropFeatureError,
    assign_relative_teams,
    extract_kit_feature,
    select_sample,
)


def load_detection_context(path: Path | None) -> dict[str, tuple[float | None, float | None]]:
    """Read coordinates only to annotate field context; no track ID is used."""
    if path is None:
        return {}
    contexts: dict[str, tuple[float | None, float | None]] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            frame = json.loads(line)
            for detection in frame.get("detections", []):
                detection_id = detection.get("detection_id")
                if detection_id:
                    contexts[str(detection_id)] = (detection.get("x"), detection.get("z"))
    return contexts


def field_context(
    x: float | None,
    z: float | None,
    field_length: float | None,
    field_width: float | None,
    *,
    edge_band_m: float = 1.0,
) -> str:
    """Annotate geometry while preserving every observation."""
    if x is None or z is None or field_length is None or field_width is None:
        return "unknown"
    dx = abs(float(x)) - field_length / 2.0
    dz = abs(float(z)) - field_width / 2.0
    if dx > 0.0 or dz > 0.0:
        return "outside"
    if min(-dx, -dz) <= edge_band_m:
        return "near_boundary"
    return "inside"


def _load_field_size(
    match_json: Path | None,
    field_length: float | None,
    field_width: float | None,
) -> tuple[float | None, float | None]:
    if match_json is not None and match_json.exists():
        match = json.loads(match_json.read_text(encoding="utf-8"))
        calibration = match.get("calibration", {})
        field_length = field_length or calibration.get("field_length")
        field_width = field_width or calibration.get("field_width")
    return (
        float(field_length) if field_length is not None else None,
        float(field_width) if field_width is not None else None,
    )


def load_crop_records(
    crop_index_path: Path,
    *,
    detections_path: Path | None = None,
    match_json: Path | None = None,
    field_length: float | None = None,
    field_width: float | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Load pixels and observation metadata; intentionally discard track IDs."""
    crop_index_path = crop_index_path.resolve()
    index = json.loads(crop_index_path.read_text(encoding="utf-8"))
    tracking_root = crop_index_path.parents[2]
    field_length, field_width = _load_field_size(match_json, field_length, field_width)
    contexts = load_detection_context(detections_path)
    records: list[dict[str, Any]] = []
    statuses = Counter()

    for raw in index.get("records", []):
        status = str(raw.get("status", "missing_status"))
        statuses[status] += 1
        if status != "saved" or not raw.get("crop_ref"):
            continue
        crop_path = (tracking_root / str(raw["crop_ref"])).resolve()
        if not crop_path.is_file():
            statuses["missing_crop_file"] += 1
            continue

        # Detection IDs are used only to join geometric context. Tracklet IDs
        # are never copied into the V2 record or passed to team discovery.
        detection_id = str(raw.get("detection_id", ""))
        x, z = contexts.get(detection_id, (None, None))
        records.append(
            {
                "image_path": crop_path,
                "timestamp_s": float(raw.get("timestamp_s") or 0.0),
                "frame_index": int(raw.get("frame_index") or 0),
                "representative_view": str(raw.get("representative_view") or "unknown"),
                "crop_width_px": int(raw.get("crop_width_px") or 0),
                "crop_height_px": int(raw.get("crop_height_px") or 0),
                "field_context": field_context(x, z, field_length, field_width),
                "world_x": x,
                "world_z": z,
            }
        )
    return records, dict(statuses)


def _sample_id(index: int) -> str:
    return f"S{index + 1:03d}"


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_gallery(path: Path, rows: list[dict[str, Any]]) -> None:
    sections = {TEAM_A: [], TEAM_B: [], UNKNOWN: []}
    for row in rows:
        sections[row["predicted_team"]].append(row)
    body: list[str] = []
    for label in (TEAM_A, TEAM_B, UNKNOWN):
        cards = []
        for row in sections[label]:
            sample_id = html.escape(row["sample_id"])
            cards.append(
                f'<article class="sample" data-id="{sample_id}" data-field-context="{html.escape(str(row["field_context"]))}" data-view="{html.escape(str(row["representative_view"]))}" data-scale-bin="{html.escape(str(row["scale_bin"]))}">'
                f'<img loading="lazy" src="{html.escape(row["crop_file"])}" alt="Recorte {sample_id}">'
                f'<h3>{sample_id} · predicción {html.escape(label)}</h3>'
                f'<p>{float(row["timestamp_s"]):.2f} s · {html.escape(row["representative_view"])} · '
                f'{row["crop_width_px"]}×{row["crop_height_px"]} px · '
                f'{html.escape(row["field_context"])}</p>'
                f'<label>Referencia visual '
                f'<select class="manual-team"><option value="">Sin etiquetar</option>'
                f'<option>{TEAM_A}</option><option>{TEAM_B}</option><option>{UNKNOWN}</option></select></label>'
                f'<label>Luz <select class="lighting"><option value="">Desconocida</option>'
                f'<option>sun</option><option>shade</option><option>mixed</option></select></label>'
                f'<label>Rol <select class="role"><option value="">Desconocido</option>'
                f'<option>player</option><option>referee</option><option>goalkeeper</option>'
                f'<option>staff_or_spectator</option></select></label>'
                f'<textarea class="notes" placeholder="Notas de revisión"></textarea>'
                f'</article>'
            )
        body.append(
            f'<section><h2>{html.escape(label)} <span>({len(cards)})</span></h2>'
            f'<div class="grid">{"".join(cards)}</div></section>'
        )

    page = """<!doctype html>
<html lang="es"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Tracking V2 · Team Discovery</title>
<style>
body{font:15px system-ui,sans-serif;margin:24px;background:#f5f6f8;color:#18202a}
h1{margin-bottom:4px}.notice{max-width:900px;color:#425466}
section{margin:28px 0}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:12px}
.sample{background:white;border:1px solid #d9dfe6;border-radius:8px;padding:10px;display:flex;flex-direction:column;gap:6px}
.sample img{height:190px;max-width:100%;object-fit:contain;background:#e9edf1}
.sample h3{font-size:14px;margin:0}.sample p{font-size:12px;margin:0;color:#4e5b68}
label{font-size:12px;display:grid;grid-template-columns:100px 1fr;gap:6px;align-items:center}
select,textarea{font:inherit;max-width:100%;padding:4px}textarea{height:42px}
button{padding:9px 14px;margin:8px 6px 14px 0}.actions{position:sticky;top:0;background:#f5f6f8;padding:8px 0;z-index:2}
</style>
<h1>Tracking V2 · Team Discovery</h1>
<p class="notice">Agrupaciones previstas por el experimento; no son etiquetas de verdad. Revisa los recortes y usa “Exportar etiquetas” para crear una referencia visual independiente. Los IDs de tracklet no se muestran.</p>
<div class="actions"><button id="save">Guardar en este navegador</button><button id="export">Exportar etiquetas CSV</button></div>
""" + "".join(body) + """
<script>
const cards=[...document.querySelectorAll('.sample')];
const key=c=>'tracking-v2-label-'+c.dataset.id;
function restore(){for(const c of cards){const d=JSON.parse(localStorage.getItem(key(c))||'{}');for(const cls of ['manual-team','lighting','role','notes']){const e=c.querySelector('.'+cls);if(e)e.value=d[cls]||''}}}
function save(){for(const c of cards){const d={};for(const cls of ['manual-team','lighting','role','notes']){const e=c.querySelector('.'+cls);d[cls]=e?e.value:''}localStorage.setItem(key(c),JSON.stringify(d))}}
document.getElementById('save').onclick=()=>{save();alert('Etiquetas guardadas en este navegador.')};
document.getElementById('export').onclick=()=>{save();const q=x=>'"'+String(x??'').replaceAll('"','""')+'"';const lines=[['sample_id','manual_team','lighting','role','field_context','representative_view','scale_bin','review_notes'].map(q).join(',')];for(const c of cards){const d=JSON.parse(localStorage.getItem(key(c))||'{}');lines.push([c.dataset.id,d['manual-team'],d.lighting,d.role,c.dataset.fieldContext,c.dataset.view,c.dataset.scaleBin,d.notes].map(q).join(','))}const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([lines.join('\\n')],{type:'text/csv'}));a.download='manual_labels.csv';a.click();URL.revokeObjectURL(a.href)};
restore();
</script></html>"""
    path.write_text(page, encoding="utf-8")


def run_experiment(
    crop_index: Path,
    output_dir: Path,
    *,
    detections: Path | None = None,
    match_json: Path | None = None,
    field_length: float | None = None,
    field_width: float | None = None,
    sample_count: int = 96,
    outside_count: int = 12,
    seed: int = 17,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    records, input_statuses = load_crop_records(
        crop_index,
        detections_path=detections,
        match_json=match_json,
        field_length=field_length,
        field_width=field_width,
    )
    selected = select_sample(
        records, sample_count=sample_count, outside_count=outside_count, seed=seed
    )
    sample_ids = [_sample_id(i) for i in range(len(selected))]
    features = []
    feature_errors: list[str | None] = []
    for record in selected:
        try:
            features.append(extract_kit_feature(record["image_path"]))
            feature_errors.append(None)
        except (OSError, CropFeatureError, ValueError) as error:
            features.append(None)
            feature_errors.append(str(error))

    predictions = assign_relative_teams(features, seed=seed, sample_ids=sample_ids)
    crop_dir = output_dir / "crops"
    crop_dir.mkdir()
    assignment_rows: list[dict[str, Any]] = []
    manual_rows: list[dict[str, Any]] = []
    gallery_rows: list[dict[str, Any]] = []
    for i, (record, prediction) in enumerate(zip(selected, predictions, strict=True)):
        sample_id = sample_ids[i]
        crop_file = f"crops/{sample_id}.jpg"
        shutil.copyfile(record["image_path"], output_dir / crop_file)
        scale = "small" if record["crop_height_px"] < 40 else "medium" if record["crop_height_px"] < 80 else "large"
        row = {
            "sample_id": sample_id,
            "timestamp_s": f"{record['timestamp_s']:.4f}",
            "representative_view": record["representative_view"],
            "crop_width_px": record["crop_width_px"],
            "crop_height_px": record["crop_height_px"],
            "scale_bin": scale,
            "field_context": record["field_context"],
            "world_x": "" if record["world_x"] is None else f"{float(record['world_x']):.4f}",
            "world_z": "" if record["world_z"] is None else f"{float(record['world_z']):.4f}",
            "predicted_team": prediction["team"],
            "confidence": f"{prediction['confidence']:.4f}",
            "reason": prediction["reason"],
            "crop_file": crop_file,
            "feature_error": feature_errors[i] or "",
        }
        assignment_rows.append(row)
        manual_rows.append(
            {
                "sample_id": sample_id,
                "manual_team": "",
                "lighting": "",
                "role": "",
                "field_context": record["field_context"],
                "representative_view": record["representative_view"],
                "scale_bin": scale,
                "review_notes": "",
            }
        )
        gallery_rows.append(row)

    assignment_fields = list(assignment_rows[0]) if assignment_rows else ["sample_id", "predicted_team"]
    _write_csv(output_dir / "assignments.csv", assignment_fields, assignment_rows)
    _write_csv(
        output_dir / "labels.csv",
        ["sample_id", "manual_team", "lighting", "role", "field_context", "representative_view", "scale_bin", "review_notes"],
        manual_rows,
    )
    _write_gallery(output_dir / "index.html", gallery_rows)
    return {
        "method": "upper-torso chromaticity and hue histogram; deterministic two-means; relative rejection to UNKNOWN/SPECIAL",
        "sample_count": len(assignment_rows),
        "requested_sample_count": sample_count,
        "requested_outside_sample_count": outside_count,
        "field_context_counts": dict(Counter(row["field_context"] for row in assignment_rows)),
        "predicted_counts": dict(Counter(row["predicted_team"] for row in assignment_rows)),
        "source_crop_records": len(records),
        "input_crop_status_counts": input_statuses,
        "seed": seed,
        "manual_reference_labels": 0,
        "metrics": None,
        "limitations": [
            "source crop index was selected by a legacy V1 quality/tracklet process; V2 ignores its tracklet IDs but cannot remove this sample-selection bias",
            "only saved crops can be analyzed; raw candidate detections without crops are not assigned",
            "outside means outside the calibrated rectangle; it does not establish whether an observation is a player, staff member, or spectator",
            "lighting and role metrics remain pending independent visual labels",
        ],
        "output_dir": str(output_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small offline relative team-discovery experiment.")
    parser.add_argument("--crop-index", type=Path, required=True)
    parser.add_argument("--detections", type=Path)
    parser.add_argument("--match-json", type=Path)
    parser.add_argument("--field-length", type=float)
    parser.add_argument("--field-width", type=float)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=96)
    parser.add_argument("--outside-count", type=int, default=12)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    summary = run_experiment(
        args.crop_index,
        args.output_dir,
        detections=args.detections,
        match_json=args.match_json,
        field_length=args.field_length,
        field_width=args.field_width,
        sample_count=args.sample_count,
        outside_count=args.outside_count,
        seed=args.seed,
    )
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Gallery: {args.output_dir / 'index.html'}")
    print(f"Manual labels: {args.output_dir / 'labels.csv'}")


if __name__ == "__main__":
    main()
