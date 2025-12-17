import pandas as pd
from pathlib import Path
import sys
from tqdm import tqdm 

def pq2jsonl(src: Path, dst: Path) -> None:
    df = pd.read_parquet(src)
    df.to_json(dst, orient='records', lines=True, force_ascii=False)

def main():
    in_root  = Path('/home/ma-user/work/cache/data/fineweb-edu-100b-parquet/')
    out_root = Path('/home/ma-user/work/cache/data/fineweb-edu-100b-jsonl/')
    
    all_parquets = sorted(in_root.rglob("*.parquet"))
    for pq_path in tqdm(all_parquets, desc="parquet→jsonl"):
        rel_path = pq_path.relative_to(in_root).with_suffix(".jsonl")
        out_path = out_root / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        print(pq_path)

        pq2jsonl(pq_path, out_path)

if __name__ == "__main__":
    main()