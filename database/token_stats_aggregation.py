from collections.abc import Mapping
import unicodedata


TOKEN_FIELDS = (
    "prompt_tokens",
    "completion_tokens",
    "prompt_cache_hit_tokens",
    "prompt_cache_miss_tokens",
    "reasoning_tokens",
)
LEGACY_PROVIDER_LABEL = "legacy/unknown"


def _clean_model_name(value: object) -> str:
    return unicodedata.normalize("NFKC", str(value or "")).strip()


def _model_key(value: object) -> str:
    return _clean_model_name(value).casefold()


def _empty_totals() -> dict[str, int]:
    return {field: 0 for field in TOKEN_FIELDS}


def _format_totals(totals: Mapping[str, int]) -> dict[str, int | float]:
    prompt = int(totals.get("prompt_tokens", 0) or 0)
    completion = int(totals.get("completion_tokens", 0) or 0)
    cache_hit = int(totals.get("prompt_cache_hit_tokens", 0) or 0)
    cache_miss = int(totals.get("prompt_cache_miss_tokens", 0) or 0)
    cache_total = cache_hit + cache_miss
    return {
        "prompt": prompt,
        "completion": completion,
        "reasoning": int(totals.get("reasoning_tokens", 0) or 0),
        "cache_hit": cache_hit,
        "cache_miss": cache_miss,
        "cache_hit_ratio": cache_hit / cache_total if cache_total else 0.0,
        "total": prompt + completion,
    }


def merge_token_stats_by_model(
    aggregate: Mapping[tuple[str, str], Mapping[str, int]],
) -> list[dict]:
    """Merge exact normalized model names while retaining dynamic provider totals."""

    models: dict[str, dict] = {}
    for (raw_model, raw_provider), totals in aggregate.items():
        model_name = _clean_model_name(raw_model)
        key = _model_key(model_name)
        provider = str(raw_provider or "").strip() or LEGACY_PROVIDER_LABEL
        entry = models.setdefault(
            key,
            {
                "model_names": set(),
                "totals": _empty_totals(),
                "providers": {},
            },
        )
        entry["model_names"].add(model_name)
        provider_totals = entry["providers"].setdefault(provider, _empty_totals())
        for field in TOKEN_FIELDS:
            value = int(totals.get(field, 0) or 0)
            entry["totals"][field] += value
            provider_totals[field] += value

    rows = []
    for key in sorted(models):
        entry = models[key]
        model_variants = sorted(
            entry["model_names"],
            key=lambda value: (value.casefold(), value),
        )
        providers = sorted(entry["providers"])
        provider_breakdown = [
            {
                "provider": provider,
                **_format_totals(entry["providers"][provider]),
            }
            for provider in providers
        ]
        rows.append(
            {
                "model": model_variants[0] if model_variants else "",
                "model_variants": model_variants,
                "provider": providers[0] if len(providers) == 1 else "mixed",
                "providers": providers,
                "provider_breakdown": provider_breakdown,
                **_format_totals(entry["totals"]),
            }
        )
    return rows
