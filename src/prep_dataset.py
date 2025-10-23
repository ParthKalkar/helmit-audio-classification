import argparse, os, csv, glob

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="Data root folder containing class subdirs")
    ap.add_argument("--harmful_dir", type=str, default="harmful")
    ap.add_argument("--safe_dir", type=str, default="safe")
    ap.add_argument("--out_csv", type=str, default="./data/labels.csv")
    args = ap.parse_args()

    harmful_glob = glob.glob(os.path.join(args.root, args.harmful_dir, "**/*.mp3"), recursive=True)
    safe_glob    = glob.glob(os.path.join(args.root, args.safe_dir, "**/*.mp3"), recursive=True)

    rows = []
    for p in harmful_glob:
        rel = os.path.relpath(p, args.root)
        rows.append((rel, 1))
    for p in safe_glob:
        rel = os.path.relpath(p, args.root)
        rows.append((rel, 0))

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["filepath","label"]); w.writerows(rows)

    print(f"Wrote {args.out_csv}: {len(rows)} rows")

if __name__ == "__main__":
    main()
