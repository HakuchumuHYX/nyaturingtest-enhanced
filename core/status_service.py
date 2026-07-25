from ..config import (
    get_chat_thinking_settings,
    get_config_load_status,
    get_effective_vlm_mode,
    get_vision_settings,
    should_use_standalone_vlm,
)
from .metrics import metrics


class StatusService:
    """Build a session/runtime diagnostic view independent of NoneBot matchers."""

    async def describe(self, state) -> str:
        async with state.session_lock:
            await state.session.load_session()
            status = state.session.status()
        chat_thinking = get_chat_thinking_settings()
        chat_vision = get_vision_settings("chat")
        feedback_vision = get_vision_settings("feedback")
        lines = [
            "",
            "Provider:",
            (
                "- Chat thinking: "
                f"{'on' if chat_thinking.get('enabled') else 'off'} "
                f"{chat_thinking.get('reasoning_effort', '')}"
            ).strip(),
            "- Feedback thinking: off (fixed for deterministic analysis)",
            (
                f"- Vision: chat={'native' if chat_vision['enabled'] else 'text'}, "
                f"feedback={'native' if feedback_vision['enabled'] else 'text'}, "
                f"vlm_mode={get_effective_vlm_mode()}, "
                f"standalone={'on' if should_use_standalone_vlm() else 'off'}"
            ),
            f"- Queue length: {len(state.messages_chunk)}",
            (
                f"- Metrics: llm={metrics.llm_success}/{metrics.llm_failure}, "
                f"vlm={metrics.vlm_success}/{metrics.vlm_failure}, "
                f"db_write_failure={metrics.db_write_failure}"
            ),
        ]
        for name, client in (
            ("Chat", state.client),
            ("Feedback", state.feedback_client),
        ):
            provider_status = getattr(client, "provider_status", None)
            if provider_status and provider_status.last_error_type:
                lines.append(
                    f"- {name} last_error={provider_status.last_error_type} "
                    "circuit_remaining="
                    f"{provider_status.circuit_remaining_seconds}s"
                )
        config_status = get_config_load_status()
        if not config_status.ok or config_status.source != "file":
            lines.append(
                f"- Config: source={config_status.source} "
                f"ok={config_status.ok} error={config_status.error_type}"
            )
        return status + "\n".join(lines)
