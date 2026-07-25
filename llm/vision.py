from dataclasses import dataclass, field, replace


VISION_DETAILS = {"low", "high", "auto"}


@dataclass(frozen=True)
class VisionInput:
    """Ephemeral image input for an OpenAI-compatible multimodal request."""

    ref_id: str
    data_url: str = field(repr=False)
    is_sticker: bool = False
    source: str = "primary"
    detail: str = "auto"

    def with_detail(self, detail: str) -> "VisionInput":
        normalized = str(detail or "auto").strip().lower()
        if normalized not in VISION_DETAILS:
            normalized = "auto"
        return replace(self, detail=normalized)

    def to_openai_content(self) -> list[dict]:
        source_label = "引用消息图片" if self.source == "referenced" else "当前消息图片"
        return [
            {
                "type": "text",
                "text": f"[{source_label} image_ref={self.ref_id}]",
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": self.data_url,
                    "detail": self.detail,
                },
            },
        ]
