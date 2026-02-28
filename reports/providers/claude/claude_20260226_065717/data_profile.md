# Data Profile Report

## Dataset Overview
- **Total Rows:** 5,589
- **Source:** ChatGPT Export
- **Primary Language:** en (54.2% confidence)

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
    "parent": "NoneType",
    "children": "list",
    "raw_bytes": "int",
    "char_count": "int"
  },
  "nullable_fields": [
    "parent"
  ]
}
```

## Splits

- **user:** 2,855 (51.1%)
- **assistant:** 2,734 (48.9%)

## Diagnostics
- **Malformed Rows:** 0 (0.0%)

### Null Rates by Field

- `parent`: 13.5%