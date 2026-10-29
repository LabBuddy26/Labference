from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile

from eai4s_eval.benchmark_catalog import ATOMIC_ACTIONS, CATALOG_PATH, validate_benchmark_catalog


SHEET_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
REL_NS = {
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "p": "http://schemas.openxmlformats.org/package/2006/relationships",
}

TRACKS = {
    "长程任务_12": {"track_id": "long_horizon", "label": "长程任务", "prefix": "L", "description": "包含状态依赖、顺序约束和多阶段执行的长程任务"},
    "操作中推理_12": {"track_id": "reasoning_in_action", "label": "操作中推理", "prefix": "R", "description": "根据视觉或结构化状态进行判断并选择后续动作"},
}

# Implemented benchmark tasks and their eval task specs (B* yaml files).
# Keep in sync with eai4s_eval/configs/tasks/ and BENCHMARK_TASK_BINDINGS.
IMPLEMENTED_TASK_SPECS = {
    "L01": "../tasks/B11_level2_sampletransport.yaml",
    "L02": "../tasks/B16_level2_sample_organization.yaml",
    "L03": "../tasks/B30_level2_L03_singlepipetting.yaml",
    "L04": "../tasks/B19_level4_liquid_mixing.yaml",
    "L05": "../tasks/B31_level2_L05_formulation_spotting.yaml",
    "L06": "../tasks/B17_level2_scale_weights.yaml",
    "L07": "../tasks/B14_level2_centrifuge_load.yaml",
    "L08": "../tasks/B13_level2_nmrinjection.yaml",
    "L09": "../tasks/B12_level4_device_operation.yaml",
    "L10": "../tasks/B15_level2_cohort_analysis.yaml",
    "L11": "../tasks/B18_level4_condition_module.yaml",
    "L12": "../tasks/B20_level4_clear_reset.yaml",
    "R02": "../tasks/B21_level2_R02_wights.yaml",
    "R01": "../tasks/B28_level3_R01_scaledigital.yaml",
    "R03": "../tasks/B25_level3_R03_temperaturepanel.yaml",
    "R04": "../tasks/B29_level3_R04_centrifugepanel.yaml",
    "R05": "../tasks/B26_level3_R05_devicestatus.yaml",
    "R06": "../tasks/B32_level2_R06_centrifuge_opposite_slot.yaml",
    "R07": "../tasks/B33_level2_R07_centrifuge_symmetric_pair.yaml",
    "R08": "../tasks/B23_level2_R08_nmrsample.yaml",
    "R09": "../tasks/B24_level2_R09_nmrrack.yaml",
    "R10": "../tasks/B22_level3_R10_rackplace.yaml",
    "R11": "../tasks/B27_level2_R11_plateposition.yaml",
    "R12": "../tasks/B34_level3_R12_devicecover_status.yaml",
}

ATOMIC_BY_TASK = {
    "L01": ["grasp", "lift", "move", "place"], "L02": ["grasp", "move", "place"],
    "L03": ["grasp", "move", "press", "place"], "L04": ["grasp", "move", "press", "shake", "place"],
    "L05": ["grasp", "move", "press", "place"], "L06": ["open", "grasp", "move", "place", "close"],
    "L07": ["open", "grasp", "move", "place", "close", "press"],
    "L08": ["grasp", "lift", "move", "place"], "L09": ["open", "grasp", "move", "place", "close", "press"],
    "L10": ["grasp", "lift", "move", "place"], "L11": ["open", "grasp", "move", "place", "close", "press"],
    "L12": ["grasp", "move", "place", "close"],
    "R01": ["press"], "R02": ["grasp", "move", "place"], "R03": ["open", "press"],
    "R04": ["press"], "R05": ["open", "press"], "R06": ["move", "place"],
    "R07": ["grasp", "move", "place"], "R08": ["move", "place"], "R09": ["move", "place"],
    "R10": ["move", "place"], "R11": ["move"], "R12": ["close", "press"],
}


def _xlsx_rows(path: Path) -> dict[str, list[dict[str, str]]]:
    with ZipFile(path) as archive:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            shared = ["".join(node.text or "" for node in item.iterfind(".//m:t", SHEET_NS)) for item in root.findall("m:si", SHEET_NS)]
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        targets = {item.attrib["Id"]: item.attrib["Target"] for item in relationships}
        result: dict[str, list[dict[str, str]]] = {}
        sheet_root = workbook.find("m:sheets", SHEET_NS)
        if sheet_root is None:
            raise ValueError("Workbook does not contain a sheets collection")
        for sheet in sheet_root:
            name = sheet.attrib["name"]
            rel_id = sheet.attrib[f"{{{REL_NS['r']}}}id"]
            target = targets[rel_id]
            member = target if target.startswith("xl/") else "xl/" + target.lstrip("/")
            root = ET.fromstring(archive.read(member))
            matrix: list[dict[str, str]] = []
            for row in root.findall(".//m:sheetData/m:row", SHEET_NS):
                values: dict[str, str] = {}
                for cell in row.findall("m:c", SHEET_NS):
                    column = re.match(r"[A-Z]+", cell.attrib.get("r", ""))
                    if not column:
                        continue
                    value = cell.find("m:v", SHEET_NS)
                    text = value.text if value is not None and value.text is not None else ""
                    if cell.attrib.get("t") == "s" and text:
                        text = shared[int(text)]
                    values[column.group(0)] = text.strip()
                if values:
                    matrix.append(values)
            if not matrix:
                result[name] = []
                continue
            headers = matrix[0]
            result[name] = [
                {headers[column]: value for column, value in row.items() if column in headers}
                for row in matrix[1:]
            ]
        return result


def _score_components(rule: str) -> list[dict[str, object]]:
    parts = []
    for label, weight in re.findall(r"([^+]+?)(\d+)(?=\+|$)", rule.replace("（", "").replace("）", "")):
        clean_label = label.strip()
        if any(token in clean_label for token in ("完成", "终态", "答案", "动作正确")):
            metric = "success"
        elif "效率" in clean_label:
            metric = "efficiency_score_100"
        elif any(token in clean_label for token in ("力", "扭矩")):
            metric = "force_quality_score_100"
        elif any(token in clean_label for token in ("体积", "液量", "浓度", "读数")):
            metric = "quantity_quality_score_100"
        elif any(token in clean_label for token in ("顺序", "依赖")):
            metric = "sequence_quality_score_100"
        elif any(token in clean_label for token in ("证据", "视觉", "文字", "标签")):
            metric = "evidence_quality_score_100"
        elif any(token in clean_label for token in ("子目标", "孔级")):
            metric = "subgoal_quality_score_100"
        else:
            metric = "geometry_quality_score_100"
        parts.append({"component": clean_label, "weight": int(weight), "metric": metric})
    if sum(int(item["weight"]) for item in parts) != 100:
        raise ValueError(f"Cannot parse a 100-point scoring rule: {rule}")
    return parts


def build_catalog(source: Path) -> dict[str, object]:
    sheets = _xlsx_rows(source)
    tasks = []
    for sheet_name, track in TRACKS.items():
        rows = sheets.get(sheet_name)
        if rows is None:
            raise ValueError(f"Workbook is missing sheet {sheet_name!r}")
        for row in rows:
            task_id = row["任务ID"]
            tasks.append({
                "task_id": task_id,
                "track_id": track["track_id"],
                "subboard": row.get("子榜"),
                "scene": row.get("实验场景"),
                "task_name": row.get("任务内容"),
                "setting": row.get("任务设定"),
                "capability": row.get("主评能力"),
                "dependencies": row.get("步骤或依赖"),
                "minimum_assets": row.get("最小资产"),
                "simulation_mvp": row.get("Isaac Sim MVP 实现") or row.get("Isaac Sim实现"),
                "success_conditions": row.get("成功条件"),
                "scoring_rule": row.get("评分规则（100分）"),
                "score_components": _score_components(row.get("评分规则（100分）", "")),
                "failure_safety": row.get("失败/安全"),
                "atomic_actions": ATOMIC_BY_TASK[task_id],
                "status": "implemented" if task_id in IMPLEMENTED_TASK_SPECS else "planned",
                "task_spec": IMPLEMENTED_TASK_SPECS.get(task_id),
            })
    tasks.sort(key=lambda item: item["task_id"])
    catalog = {
        "schema_version": "laboratory_benchmark_catalog/v1",
        "catalog_id": "laboratory_benchmark_v1",
        "version": "2.0.0",
        "source": {"filename": source.name, "sheets": list(TRACKS)},
        "aggregation": {
            "task_success": "valid_episode_micro",
            "track_success": "evaluated_task_macro",
            "overall_success": "evaluated_task_macro",
            "atomic_success": "action_instance_micro_per_action_then_action_macro",
            "atomic_in_official_score": False,
        },
        "tracks": [{**value, "task_count": 12} for value in TRACKS.values()],
        "atomic_actions": [{"action_id": action_id, "label": label} for action_id, label in ATOMIC_ACTIONS],
        "tasks": tasks,
    }
    validate_benchmark_catalog(catalog)
    return catalog


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the versioned laboratory benchmark catalog from the task workbook.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=CATALOG_PATH)
    parser.add_argument("--check", action="store_true", help="Fail if the checked-in catalog differs from the workbook.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    catalog = build_catalog(args.source)
    rendered = json.dumps(catalog, ensure_ascii=False, indent=2) + "\n"
    if args.check:
        if not args.output.exists() or args.output.read_text(encoding="utf-8") != rendered:
            raise SystemExit(f"Catalog drift detected: rebuild {args.output}")
        print(f"Catalog is current: {args.output}")
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(f"Wrote {len(catalog['tasks'])} tasks to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
