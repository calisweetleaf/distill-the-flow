from pathlib import Path

path = Path(r"C:/Users/treyr/Desktop/Dev-Drive/distill_the_flow/tools/json_tool.py")
text = path.read_text(encoding="utf-8")

old = '''    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(merged_list, indent=2, ensure_ascii=False), encoding="utf-8")
'''

new = '''    # Write output without materializing one giant JSON string in memory.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(merged_list, fh, indent=2, ensure_ascii=False)
'''

if old not in text:
    raise SystemExit('merge write block not found')

text = text.replace(old, new)
path.write_text(text, encoding='utf-8')
print('patched merge writer in tools/json_tool.py')
