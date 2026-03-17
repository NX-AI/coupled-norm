import pandas as pd
import glob
import json

paths = [
    './multirun/2026-03-17/sweep-decode-qwen-1-7',
    './multirun/2026-03-17/sweep-decode-no-qwen-1-7',
    './multirun/2026-03-17/sweep-train',
    './multirun/2026-03-17/llama-8b-decode'
]
files = []
for p in paths:
    files.extend(glob.glob(f"{p}/**/result.json", recursive=True))
df = pd.DataFrame([json.load(open(f)) for f in files])

df["nonorm"] = df["REF"] / df["NONORM"]
df["refref"] = df["REF"] / df["REFREF"]
df["speedup_pct"] = (100 * df["REF"] / df["COUPLED"] - 100).round(2)

df = df.round(3)
df = df.sort_values(by=['model_id', 'prompt_len', 'batch_size', 'new_tokens'])

print(df)

print("\n\nOn device decode with BS=1 new_tokens=256")
print(df[(df["batch_size"] == 1) & (df["new_tokens"] == 256)])

print("\n\nLarge scale decode with BS=256 new_tokens=256")
print(df[(df["batch_size"] == 256) & (df["new_tokens"] == 256)])

print("\n\nTrain setting with BS=256 new_tokens=1")
print(df[(df["batch_size"] == 128) & (df["prompt_len"] > 100) & (df["new_tokens"] == 1)])
