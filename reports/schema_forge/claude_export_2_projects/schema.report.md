# Schema Template: `projects.json`

**Provider:** `claude_conversations`  
**File size:** 73.2 MB  
**Records sampled:** 93  
**Sampled (not full):** False  
**Max depth:** 3  
**Total fields:** 16  
**Extracted at:** 2026-02-27T18:34:34.092064  

## Structure Template

```json
[
  {
    "uuid?": "str",
    "name?": "str",
    "description?": "str",
    "is_private?": "bool",
    "is_starter_project?": "bool",
    "prompt_template?": "str",
    "created_at?": "str",
    "updated_at?": "str",
    "creator?": {
      "uuid?": "str",
      "full_name?": "str"
    },
    "docs?": [
      {
        "uuid?": "str",
        "filename?": "str",
        "content?": "str",
        "created_at?": "str"
      }
    ]
  }
]
```

## Anomalies

| Type | Path/Line | Confidence |
|------|-----------|------------|
| `empty_list` | `$[38].docs` | unknown |
| `empty_list` | `$[55].docs` | unknown |
| `empty_list` | `$[62].docs` | unknown |
| `empty_list` | `$[66].docs` | unknown |
| `empty_list` | `$[90].docs` | unknown |
| `empty_list` | `$[93].docs` | unknown |

## Provider Notes

> Anthropic Claude export - conversations.json
