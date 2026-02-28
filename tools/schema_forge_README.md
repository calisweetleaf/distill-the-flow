# Schema Forge v1.0

> Zero-data structural extractor for JSON/JSONL provider exports.  
> Memory-safe. Provider-aware. Merge-capable.

## What It Does

Given any provider data export (ChatGPT, Claude, Gemini, Perplexity), Schema Forge:

1. **Extracts the full structure** as a template — every field, every type, optionals marked with `?`, no actual data
2. **Identifies the provider** from filename + key fingerprinting
3. **Outputs three formats**: JSON template, YAML schema, Markdown report
4. **Reports anomalies**: empty arrays, malformed lines, union types
5. **Merges Claude exports**: combines your two split `conversations.json` files, deduplicating by UUID

---

## Install

```bash
pip install ijson pyyaml
# ijson is optional but strongly recommended for files > 50MB
```

---

## Usage

### Single file

```bash
python json_tool.py conversations.jsonl
python json_tool.py ./claude_export/
```

### Batch via config (recommended for Moonshine pipeline)

```bash
# Edit config.yaml with your actual paths
python json_tool.py --config config.yaml
```

### Merge two Claude conversation exports

```bash
python json_tool.py --merge jan_conversations.json aug_conversations.json \
    --merge-output ./merged/conversations.json
```

### Options

```
--output-dir DIR      Where to write files (default: ./schema_out)
--mode auto|json|jsonl  Force parse mode (default: auto-detect)
--sample-limit N      Max records to scan in huge files (default: 5000)
--no-stdout           Only write files, suppress terminal output
--verbose             Debug logging
```

---

## Output Files

For each input, written to `--output-dir`:

| File | Contents |
|------|----------|
| `*.template.json` | Full structural template, no data |
| `*.schema.yaml` | Same structure in YAML |
| `*.report.md` | Human-readable report with stats + anomalies |
| `*.stats.json` | File size, record count, depth, provider |
| `*.anomalies.json` | Structural anomalies (only if any found) |

---

## Template Format

```json
{
  "uuid": "str",
  "name": "str",
  "created_at": "str",
  "chat_messages": [
    {
      "uuid": "str",
      "text": "str",
      "sender": "str",
      "created_at": "str",
      "attachments?": "list"
    }
  ],
  "account?": {
    "uuid": "str"
  }
}
```

`?` suffix = field is optional (not present in all records).  
Union types shown as array: `["str", "null"]`

---

## Provider Fingerprinting

Automatically identifies:

| Provider | Detected From |
|----------|--------------|
| `openai_gpt` | `conversations.jsonl` + keys: `mapping`, `current_node` |
| `claude_conversations` | `conversations.json` + keys: `uuid`, `chat_messages` |
| `claude_memories` | `memories.json` + keys: `memories` |
| `claude_projects` | `projects.json` + keys: `projects` |
| `gemini` | Takeout structure + keys: `conversation`, `role` |
| `perplexity` | `history.json` + keys: `query`, `answer`, `sources` |

---

## Claude Export Merge

Your two Claude exports (Jan-Aug 2025, Aug 2025-Feb 2026) have overlapping dates around August 8. The merger:

- Loads both files
- Deduplicates conversations by `uuid`
- When duplicate found, keeps the version with **more messages**
- Sorts merged list by `created_at` ascending
- Writes a single `conversations.json`
- Reports how many duplicates were removed

---

## Large File Handling

| File size | Strategy |
|-----------|----------|
| < 50MB | Full in-memory load |
| > 50MB + ijson installed | Streaming parse, never loads full file |
| JSONL any size | Always streamed line-by-line |

`--sample-limit` caps how many records are scanned. Default 5000 is enough to capture the complete schema for any provider. For forensics work where you want full coverage, set `--sample-limit 0` (no limit — will scan everything).

---

## Moonshine Integration

Recommended workflow:

```
1. Run once per provider export to get templates
2. Commit templates to repo — they're your ground truth schemas
3. In Moonshine parser, import template as expected shape before processing
4. Point config.yaml at your actual export paths
5. Add config.yaml to your VM's startup — schema always current
```

---

## Notes

- `--sample-limit 0` = no limit (full scan, slower on huge files)
- JSONL autodetection: checks extension first, then content sniff
- Empty arrays/objects are logged as anomalies with `confidence: unknown` because the schema of their elements can't be inferred from an empty sample
