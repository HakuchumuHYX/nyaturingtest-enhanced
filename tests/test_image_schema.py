import unittest
from pathlib import Path
import sys

PLUGIN_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLUGIN_DIR))

from memory.image_schema import (
    ImageWithDescription,
    parse_vlm_response,
    render_image_text,
    merge_segment_metas,
    gif_target_count,
    PRAGMATIC_INTENTS,
)


class ImageWithDescriptionSchemaTests(unittest.TestCase):
    def test_full_schema_roundtrip(self):
        desc = ImageWithDescription(
            visual_description="绿发双马尾女孩比耶",
            ocr_text="我装的",
            entities=[{"name": "初音未来", "type": "character", "confidence": 0.85}],
            pragmatic_intent="否认",
            affect={"valence": 0.7, "arousal": 0.5, "dominance": 0.2},
            is_sticker=True,
            temporal=[{"frame": 1, "action": "举手"}],
        )
        js = desc.to_json()
        restored = ImageWithDescription.from_json(js)
        self.assertEqual("绿发双马尾女孩比耶", restored.visual_description)
        self.assertEqual("我装的", restored.ocr_text)
        self.assertEqual([{"name": "初音未来", "type": "character", "confidence": 0.85}], restored.entities)
        self.assertEqual("否认", restored.pragmatic_intent)
        self.assertEqual({"valence": 0.7, "arousal": 0.5, "dominance": 0.2}, restored.affect)
        self.assertTrue(restored.is_sticker)
        self.assertEqual([{"frame": 1, "action": "举手"}], restored.temporal)
        # 旧字段别名同步
        self.assertEqual("绿发双马尾女孩比耶", restored.description)

    def test_old_cache_json_compatible(self):
        """旧磁盘缓存只有 description/emotion/is_sticker 三个字段。"""
        old = '{"description":"一只猫趴在键盘上","emotion":"开心, 表情包, 卖萌","is_sticker":true}'
        restored = ImageWithDescription.from_json(old)
        self.assertEqual("一只猫趴在键盘上", restored.visual_description)
        self.assertEqual("一只猫趴在键盘上", restored.description)  # 别名回填
        self.assertEqual("开心, 表情包, 卖萌", restored.emotion)  # 旧字段保留
        self.assertTrue(restored.is_sticker)
        # 缺省槽
        self.assertEqual("", restored.ocr_text)
        self.assertEqual([], restored.entities)
        self.assertEqual("无", restored.pragmatic_intent)
        self.assertEqual({"valence": 0.0, "arousal": 0.0, "dominance": 0.0}, restored.affect)

    def test_to_meta_excludes_free_text(self):
        desc = ImageWithDescription(
            visual_description="画面描述",
            ocr_text="配字",
            entities=[{"name": "X", "type": "meme", "confidence": 0.5}],
            pragmatic_intent="嘲讽",
            affect={"valence": -0.3, "arousal": 0.6, "dominance": 0.1},
            is_sticker=True,
            temporal=[{"frame": 2, "action": "转头"}],
        )
        meta = desc.to_meta()
        self.assertEqual([{"name": "X", "type": "meme", "confidence": 0.5}], meta["entities"])
        self.assertEqual("配字", meta["ocr_text"])
        self.assertEqual("嘲讽", meta["pragmatic_intent"])
        self.assertEqual({"valence": -0.3, "arousal": 0.6, "dominance": 0.1}, meta["affect"])
        self.assertEqual([{"frame": 2, "action": "转头"}], meta["temporal"])
        self.assertTrue(meta["is_sticker"])
        # 不含自由文本
        self.assertNotIn("visual_description", meta)
        self.assertNotIn("description", meta)

    def test_to_meta_isolation(self):
        """to_meta 返回的容器修改不影响原对象。"""
        desc = ImageWithDescription(entities=[{"name": "A", "type": "object", "confidence": 0.1}])
        meta = desc.to_meta()
        meta["entities"].append({"name": "B", "type": "object", "confidence": 0.2})
        meta["affect"]["valence"] = 0.9
        self.assertEqual(1, len(desc.entities))
        self.assertEqual(0.0, desc.affect["valence"])


class ParseVlmResponseTests(unittest.TestCase):
    def test_parse_full_json(self):
        resp = '{"visual_description":"绿发女孩","ocr_text":"我装的","entities":[{"name":"初音未来","type":"character","confidence":0.85}],"pragmatic_intent":"否认","affect":{"valence":0.7,"arousal":0.5,"dominance":0.2}}'
        desc = parse_vlm_response(resp, is_sticker=True)
        self.assertEqual("绿发女孩", desc.visual_description)
        self.assertEqual("我装的", desc.ocr_text)
        self.assertEqual("否认", desc.pragmatic_intent)
        self.assertEqual([{"name": "初音未来", "type": "character", "confidence": 0.85}], desc.entities)
        self.assertEqual({"valence": 0.7, "arousal": 0.5, "dominance": 0.2}, desc.affect)
        self.assertTrue(desc.is_sticker)

    def test_parse_fenced_json(self):
        resp = '一些前缀\n```json\n{"visual_description":"猫","pragmatic_intent":"卖萌"}\n```\n后缀'
        desc = parse_vlm_response(resp)
        self.assertEqual("猫", desc.visual_description)
        self.assertEqual("卖萌", desc.pragmatic_intent)

    def test_parse_bare_json_in_text(self):
        resp = '我觉得 {"visual_description":"狗","ocr_text":"汪"} 是这样'
        desc = parse_vlm_response(resp)
        self.assertEqual("狗", desc.visual_description)
        self.assertEqual("汪", desc.ocr_text)

    def test_parse_complete_failure_falls_back(self):
        """完全无法解析时不抛错，退化截断文本。"""
        resp = "这不是JSON就是一段纯文本描述"
        desc = parse_vlm_response(resp, is_sticker=False)
        self.assertEqual("这不是JSON就是一段纯文本描述"[:60], desc.visual_description)
        self.assertEqual("无", desc.pragmatic_intent)
        self.assertEqual([], desc.entities)
        self.assertEqual({"valence": 0.0, "arousal": 0.0, "dominance": 0.0}, desc.affect)
        self.assertFalse(desc.is_sticker)

    def test_parse_empty_response(self):
        desc = parse_vlm_response("", is_sticker=True)
        self.assertEqual("", desc.visual_description)
        self.assertTrue(desc.is_sticker)

    def test_parse_legacy_fields(self):
        """模型用旧字段 description/emotion 也能解析。"""
        resp = '{"description":"旧描述","emotion":"开心, 表情包, 卖萌"}'
        desc = parse_vlm_response(resp)
        self.assertEqual("旧描述", desc.visual_description)

    def test_entity_normalization_filters_empty_and_clamps(self):
        resp = ('{"entities":[{"name":"","type":"character","confidence":0.5},'
                '{"name":"好角色","confidence":1.5},'
                '{"name":"低置信","confidence":-0.2},'
                '"notadict"]}')
        desc = parse_vlm_response(resp)
        names = [e["name"] for e in desc.entities]
        self.assertEqual(["好角色", "低置信"], names)
        # confidence 夹到 [0,1]
        self.assertEqual(1.0, desc.entities[0]["confidence"])
        self.assertEqual(0.0, desc.entities[1]["confidence"])
        # 未知 type 降级为 object
        self.assertEqual("object", desc.entities[0]["type"])

    def test_affect_clamp_and_nan(self):
        resp = '{"affect":{"valence":3.0,"arousal":"bad","dominance":null}}'
        desc = parse_vlm_response(resp)
        self.assertEqual(1.0, desc.affect["valence"])   # 3.0 -> 1.0
        self.assertEqual(0.0, desc.affect["arousal"])   # 'bad' -> 0.0
        self.assertEqual(0.0, desc.affect["dominance"]) # null -> 0.0

    def test_pragmatic_intent_invalid_falls_to_none(self):
        resp = '{"visual_description":"x","pragmatic_intent":"胡说八道"}'
        desc = parse_vlm_response(resp)
        self.assertEqual("无", desc.pragmatic_intent)

    def test_temporal_normalization(self):
        resp = '{"temporal":[{"frame":1,"action":"举手"},{"frame":"bad","action":""},{"action":"转头"}]}'
        desc = parse_vlm_response(resp)
        self.assertEqual([{"frame": 1, "action": "举手"}, {"frame": 0, "action": "转头"}], desc.temporal)

    def test_all_intents_in_closed_set_accepted(self):
        for intent in PRAGMATIC_INTENTS:
            desc = parse_vlm_response(f'{{"pragmatic_intent":"{intent}"}}')
            self.assertEqual(intent, desc.pragmatic_intent)


class RenderImageTextTests(unittest.TestCase):
    def test_render_sticker_with_entities(self):
        desc = ImageWithDescription(
            visual_description="绿发双马尾女孩比耶",
            ocr_text="我装的",
            entities=[{"name": "初音未来", "type": "character", "confidence": 0.85}],
            pragmatic_intent="否认",
            affect={"valence": 0.7, "arousal": 0.5, "dominance": 0.2},
            is_sticker=True,
        )
        text = render_image_text(desc, is_sticker=True)
        self.assertTrue(text.startswith("\n[表情包|"))
        self.assertTrue(text.endswith("]\n"))
        self.assertIn("实体:初音未来(0.85)", text)
        self.assertIn("配字:我装的", text)
        self.assertIn("意图:否认", text)
        self.assertIn("情感:V0.70,A0.50,D0.20", text)
        self.assertIn("画面:绿发双马尾女孩比耶", text)

    def test_render_image_without_entities(self):
        desc = ImageWithDescription(
            visual_description="橘猫趴在键盘上",
            pragmatic_intent="卖萌",
            affect={"valence": 0.6, "arousal": 0.4, "dominance": 0.1},
            is_sticker=False,
        )
        text = render_image_text(desc, is_sticker=False)
        self.assertTrue(text.startswith("\n[图片|"))
        self.assertIn("实体:", text)  # 空实体保留键
        self.assertIn("配字:", text)  # 空配字保留键

    def test_render_temporal_appended_for_gif(self):
        desc = ImageWithDescription(
            visual_description="动图",
            temporal=[{"frame": 1, "action": "举手"}, {"frame": 2, "action": "转头"}],
            is_sticker=False,
        )
        text = render_image_text(desc, is_sticker=False)
        self.assertIn("动作:1.举手;2.转头", text)

    def test_render_no_temporal_omits_action_segment(self):
        desc = ImageWithDescription(visual_description="静态图")
        text = render_image_text(desc, is_sticker=False)
        self.assertNotIn("动作:", text)

    def test_render_multiple_entities(self):
        desc = ImageWithDescription(
            entities=[
                {"name": "初音未来", "type": "character", "confidence": 0.85},
                {"name": "某meme", "type": "meme", "confidence": 0.6},
            ],
        )
        text = render_image_text(desc, is_sticker=True)
        self.assertIn("实体:初音未来(0.85),某meme(0.6)", text)


class GifTargetCountTests(unittest.TestCase):
    def test_frame_buckets(self):
        self.assertEqual(4, gif_target_count(2))
        self.assertEqual(4, gif_target_count(4))
        self.assertEqual(6, gif_target_count(5))
        self.assertEqual(6, gif_target_count(6))
        self.assertEqual(9, gif_target_count(7))
        self.assertEqual(9, gif_target_count(9))
        self.assertEqual(16, gif_target_count(10))
        self.assertEqual(16, gif_target_count(12))
        self.assertEqual(16, gif_target_count(80))

    def test_temporal_parse_and_render_end_to_end(self):
        """GIF 拼图 prompt 产出 temporal，解析+渲染端到端。"""
        resp = ('{"visual_description":"动图情节","temporal":['
                '{"frame":1,"action":"举手"},'
                '{"frame":2,"action":"转头"},'
                '{"frame":3,"action":"放下"}],'
                '"pragmatic_intent":"卖萌"}')
        desc = parse_vlm_response(resp, is_sticker=False)
        self.assertEqual(
            [{"frame": 1, "action": "举手"}, {"frame": 2, "action": "转头"}, {"frame": 3, "action": "放下"}],
            desc.temporal,
        )
        text = render_image_text(desc, is_sticker=False)
        self.assertIn("动作:1.举手;2.转头;3.放下", text)


class MergeSegmentMetasTests(unittest.TestCase):
    def _meta(self, name):
        return {"entities": [{"name": name, "type": "character", "confidence": 0.5}],
                "ocr_text": "", "pragmatic_intent": "无",
                "affect": {"valence": 0.0, "arousal": 0.0, "dominance": 0.0},
                "temporal": [], "is_sticker": False}

    def test_all_none_returns_none(self):
        self.assertIsNone(merge_segment_metas([None, None, None]))

    def test_single_primary(self):
        m = self._meta("A")
        out = merge_segment_metas([None, m, None])
        self.assertEqual({"primary": m}, out)

    def test_multiple_primary_takes_first(self):
        a, b = self._meta("A"), self._meta("B")
        out = merge_segment_metas([a, b])
        self.assertEqual({"primary": a}, out)

    def test_referenced_only(self):
        ref1, ref2 = self._meta("R1"), self._meta("R2")
        out = merge_segment_metas([None, {"referenced": [ref1, ref2]}, None])
        self.assertEqual({"referenced": [ref1, ref2]}, out)

    def test_primary_and_referenced(self):
        p = self._meta("P")
        r = self._meta("R")
        out = merge_segment_metas([p, {"referenced": [r]}])
        self.assertEqual({"primary": p, "referenced": [r]}, out)

    def test_empty_referenced_list_ignored(self):
        out = merge_segment_metas([{"referenced": []}])
        self.assertIsNone(out)


if __name__ == "__main__":
    unittest.main()
