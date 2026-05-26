from __future__ import annotations

import json
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


ROOT = Path(__file__).resolve().parents[1]


def collect(base: Path) -> dict:
    result = {}
    for event in base.glob("**/events.out.tfevents.*"):
        acc = EventAccumulator(str(event), size_guidance={"scalars": 0})
        try:
            acc.Reload()
        except Exception as exc:
            result[str(event.relative_to(ROOT))] = {"error": str(exc)}
            continue
        payload = {}
        for tag in acc.Tags().get("scalars", []):
            vals = acc.Scalars(tag)
            if vals:
                first = vals[0]
                last = vals[-1]
                payload[tag] = {
                    "first_step": int(first.step),
                    "first_value": float(first.value),
                    "last_step": int(last.step),
                    "last_value": float(last.value),
                    "points": len(vals),
                }
        result[str(event.relative_to(ROOT))] = payload
    return result


def main() -> None:
    out = {
        "perception": collect(ROOT / "detect" / "runs"),
        "sac": collect(ROOT / "runs"),
    }
    (ROOT / "report_tb_scalars.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
