from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _json_objects(text: str) -> list[Any]:
    decoder = json.JSONDecoder()
    index = 0
    objects: list[Any] = []
    while index < len(text):
        while index < len(text) and text[index].isspace():
            index += 1
        if index >= len(text):
            break
        if text[index] != "{":
            next_json = text.find("{", index + 1)
            if next_json < 0:
                break
            index = next_json
            continue
        try:
            obj, end = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            next_json = text.find("{", index + 1)
            if next_json < 0:
                break
            index = next_json
            continue
        objects.append(obj)
        index = end
    return objects


def _response_object(text: str) -> dict[str, Any]:
    for obj in reversed(_json_objects(text)):
        if isinstance(obj, dict) and {"scenario", "resolved_cards", "evidence"} <= set(obj):
            return obj
    raise ValueError("No adjudication response JSON object found.")


def _resolved_card_ids(response: dict[str, Any]) -> set[int]:
    ids: set[int] = set()
    for resolved in response.get("resolved_cards") or []:
        for match in resolved.get("matches") or []:
            card_id = match.get("card_id")
            if isinstance(card_id, int):
                ids.add(card_id)
    return ids


def _evidence_ids(response: dict[str, Any]) -> set[str]:
    return {
        str(item.get("evidence_id"))
        for item in response.get("evidence") or []
        if item.get("evidence_id")
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_path")
    parser.add_argument("--expected")
    args = parser.parse_args()

    output_path = Path(args.output_path)
    response = _response_object(output_path.read_text(encoding="utf-8"))
    resolved_ids = _resolved_card_ids(response)
    evidence_ids = _evidence_ids(response)
    coverage = response.get("agent_coverage") or {}

    print(f"Resolved card ids: {', '.join(str(i) for i in sorted(resolved_ids)) or '(none)'}")
    print(f"Evidence count: {len(evidence_ids)}")
    print(f"Coverage sufficient: {coverage.get('sufficient')}")
    if response.get("warnings"):
        print("Warnings:")
        for warning in response["warnings"]:
            print(f"  - {warning}")

    failed = False
    if args.expected:
        expected = json.loads(Path(args.expected).read_text(encoding="utf-8"))
        for card_id in expected.get("expected_card_ids", []):
            if int(card_id) in resolved_ids:
                print(f"[OK] card_id {card_id}")
            else:
                print(f"[MISS] card_id {card_id}")
                failed = True
        for evidence_id in expected.get("expected_evidence_ids", []):
            if str(evidence_id) in evidence_ids:
                print(f"[OK] evidence {evidence_id}")
            else:
                print(f"[MISS] evidence {evidence_id}")
                failed = True
        if "require_sufficient" in expected:
            required = bool(expected["require_sufficient"])
            actual = bool(coverage.get("sufficient"))
            if actual == required:
                print(f"[OK] coverage sufficient == {required}")
            else:
                print(f"[MISS] coverage sufficient == {required} (actual {actual})")
                failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
