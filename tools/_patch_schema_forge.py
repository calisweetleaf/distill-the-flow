from pathlib import Path

path = Path(r"C:/Users/treyr/Desktop/Dev-Drive/distill_the_flow/tools/json_tool.py")
text = path.read_text(encoding="utf-8")

old_stream = '''    def _parse_json_streaming(self, file_path: Path) -> Optional[SchemaNode]:
        """Stream-parse a large JSON array."""
        root_schema = SchemaNode()
        root_schema.occurrence_count = 0
        count = 0

        try:
            for i, item in stream_json_array(file_path):
                if isinstance(item, Exception):
                    self.anomalies.append({
                        "type": "parse_error",
                        "index": i,
                        "error": str(item),
                        "confidence": "confirmed"
                    })
                    continue
                node = self._analyze(item, f"$[{i}]", depth=0)
                root_schema.merge(node, track_presence=True)
                count += 1
                if count >= self.sample_limit:
                    log.warning(f"  Sample limit ({self.sample_limit}) reached - partial schema")
                    self.stats["sampled"] = True
                    break
        except Exception as e:
            log.error(f"  Streaming parse failed: {e}")
            self.anomalies.append({"type": "streaming_error", "error": str(e), "confidence": "confirmed"})
            return None

        self.stats["record_count"] = count
        if count == 0:
            root_schema.types.add(TypeToken.LIST)
        return root_schema
'''

new_stream = '''    def _parse_json_streaming(self, file_path: Path) -> Optional[SchemaNode]:
        """Stream-parse a large JSON array."""
        root_schema = SchemaNode()
        root_schema.types.add(TypeToken.LIST)
        root_schema.occurrence_count = 1
        elem_schema = SchemaNode()
        count = 0

        try:
            for i, item in stream_json_array(file_path):
                if isinstance(item, Exception):
                    self.anomalies.append({
                        "type": "parse_error",
                        "index": i,
                        "error": str(item),
                        "confidence": "confirmed"
                    })
                    continue
                node = self._analyze(item, f"$[{i}]", depth=0)
                elem_schema.merge(node, track_presence=False)
                count += 1
                if self.sample_limit > 0 and count >= self.sample_limit:
                    log.warning(f"  Sample limit ({self.sample_limit}) reached - partial schema")
                    self.stats["sampled"] = True
                    break
        except Exception as e:
            log.error(f"  Streaming parse failed: {e}")
            self.anomalies.append({"type": "streaming_error", "error": str(e), "confidence": "confirmed"})
            if count == 0:
                self.stats["record_count"] = 0
                return None
            self.stats["partial_streaming_error"] = str(e)
            self.stats["sampled"] = True

        self.stats["record_count"] = count
        if count > 0:
            root_schema.list_element = elem_schema
        return root_schema
'''

old_jsonl = '            if valid_lines >= self.sample_limit:\n                log.warning(f"  Sample limit ({self.sample_limit}) reached at line {line_no}")\n                self.stats["sampled"] = True\n                break\n'
new_jsonl = '            if self.sample_limit > 0 and valid_lines >= self.sample_limit:\n                log.warning(f"  Sample limit ({self.sample_limit}) reached at line {line_no}")\n                self.stats["sampled"] = True\n                break\n'

old_array = '            if i >= self.sample_limit:\n                self.stats["sampled"] = True\n                break\n'
new_array = '            if self.sample_limit > 0 and (i + 1) >= self.sample_limit:\n                self.stats["sampled"] = True\n                break\n'

if old_stream not in text:
    raise SystemExit("streaming block not found")
text = text.replace(old_stream, new_stream, 1)

if old_jsonl not in text:
    raise SystemExit("jsonl limit block not found")
text = text.replace(old_jsonl, new_jsonl, 1)

if old_array not in text:
    raise SystemExit("array limit block not found")
text = text.replace(old_array, new_array, 1)

path.write_text(text, encoding="utf-8")
print("patched json_tool.py")
