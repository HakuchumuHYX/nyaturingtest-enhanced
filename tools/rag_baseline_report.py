#!/usr/bin/env python3
"""Summarize RAG baseline metrics from nyaturingtest structured logs."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import sys
from typing import Any


EVENT_NAMES = {"rag_search", "rag_prompt_budget"}
LOG_JSON_RE = re.compile(r"\|\s*(\{.*\})\s*$")
LOG_TIME_RE = re.compile(r"^(?P<month>\d{2})-(?P<day>\d{2}) (?P<clock>\d{2}:\d{2}:\d{2})")


def _percentile(values: list[float], ratio: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = int((len(ordered) - 1) * ratio + 0.5)
    index = max(0, min(len(ordered) - 1, index))
    return ordered[index]


def _number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_log_event(line: str, *, year: int) -> dict[str, Any] | None:
    match = LOG_JSON_RE.search(line)
    if not match:
        return None
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    if payload.get("event") not in EVENT_NAMES:
        return None

    time_match = LOG_TIME_RE.match(line)
    if time_match:
        timestamp = datetime.strptime(
            f"{year}-{time_match.group('month')}-{time_match.group('day')} {time_match.group('clock')}",
            "%Y-%m-%d %H:%M:%S",
        )
        payload["_timestamp"] = timestamp.isoformat()
        payload["_date"] = timestamp.date().isoformat()
    return payload


def load_events(paths: list[Path], *, year: int) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                event = parse_log_event(line, year=year)
                if event:
                    event["_path"] = str(path)
                    events.append(event)
    return events


def _field_summary(events: list[dict[str, Any]], field: str) -> dict[str, float | None]:
    values = [_number(event.get(field)) for event in events]
    numbers = [value for value in values if value is not None]
    return {
        "min": min(numbers) if numbers else None,
        "p50": _percentile(numbers, 0.50),
        "p90": _percentile(numbers, 0.90),
        "p95": _percentile(numbers, 0.95),
        "max": max(numbers) if numbers else None,
    }


def build_report(
    paths: list[Path],
    *,
    year: int,
    min_days: int = 3,
    min_sessions: int = 2,
) -> dict[str, Any]:
    events = load_events(paths, year=year)
    search_events = [event for event in events if event.get("event") == "rag_search"]
    prompt_events = [event for event in events if event.get("event") == "rag_prompt_budget"]
    dates = sorted({event["_date"] for event in events if event.get("_date")})
    sessions = sorted({str(event.get("session_id")) for event in events if event.get("session_id")})
    first_ts = min((event.get("_timestamp") for event in events if event.get("_timestamp")), default=None)
    last_ts = max((event.get("_timestamp") for event in events if event.get("_timestamp")), default=None)

    ready = (
        len(dates) >= min_days
        and len(sessions) >= min_sessions
        and bool(search_events)
        and bool(prompt_events)
    )
    missing = []
    if len(dates) < min_days:
        missing.append(f"need {min_days} dates, got {len(dates)}")
    if len(sessions) < min_sessions:
        missing.append(f"need {min_sessions} sessions, got {len(sessions)}")
    if not search_events:
        missing.append("no rag_search events")
    if not prompt_events:
        missing.append("no rag_prompt_budget events")

    return {
        "ready": ready,
        "missing": missing,
        "input_paths": [str(path) for path in paths],
        "first_timestamp": first_ts,
        "last_timestamp": last_ts,
        "dates": dates,
        "days_covered": len(dates),
        "session_ids": sessions,
        "session_count": len(sessions),
        "rag_search_count": len(search_events),
        "rag_prompt_budget_count": len(prompt_events),
        "candidate_count": _field_summary(search_events, "candidate_count"),
        "returned_count": _field_summary(search_events, "returned_count"),
        "injected_count": _field_summary(search_events, "injected_count"),
        "injected_chars": _field_summary(search_events, "injected_chars"),
        "adjusted_score_min": _field_summary(search_events, "adjusted_score_min"),
        "adjusted_score_p50": _field_summary(search_events, "adjusted_score_p50"),
        "adjusted_score_p90": _field_summary(search_events, "adjusted_score_p90"),
        "adjusted_score_max": _field_summary(search_events, "adjusted_score_max"),
        "chat_prompt_total_chars": _field_summary(prompt_events, "chat_prompt_total_chars"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path, help="Log files to scan")
    parser.add_argument("--year", type=int, default=datetime.now().year, help="Year for MM-DD log timestamps")
    parser.add_argument("--min-days", type=int, default=3)
    parser.add_argument("--min-sessions", type=int, default=2)
    parser.add_argument("--fail-if-not-ready", action="store_true")
    args = parser.parse_args(argv)

    report = build_report(
        args.logs,
        year=args.year,
        min_days=args.min_days,
        min_sessions=args.min_sessions,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    if args.fail_if_not_ready and not report["ready"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
