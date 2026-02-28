# Data Profile Report

## Dataset Overview
- **Total Rows:** 169,397
- **Source:** ChatGPT Export
- **Primary Language:** other (50.0% confidence)

## Schema
```json
{
  "fields": [
    "sample_id",
    "conversation_id",
    "role",
    "text",
    "create_time",
    "parent",
    "children",
    "raw_bytes",
    "char_count"
  ],
  "field_types": {
    "sample_id": "str",
    "conversation_id": "str",
    "role": "str",
    "text": "str",
    "create_time": "float",
    "parent": "str",
    "children": "list",
    "raw_bytes": "int",
    "char_count": "int"
  },
  "nullable_fields": []
}
```

## Splits

- **user:** 72,930 (43.1%)
- **assistant:** 76,379 (45.1%)
- **tool:** 20,088 (11.9%)

## Diagnostics
- **Malformed Rows:** 0 (0.0%)

### Null Rates by Field
