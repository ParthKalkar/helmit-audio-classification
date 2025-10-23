import os
import argparse
import gdown

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder_id", required=True, help="Google Drive folder ID")
    ap.add_argument("--out_dir", default="./data")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    url = f"https://drive.google.com/drive/folders/{args.folder_id}"
    gdown.download_folder(url=url, output=args.out_dir, remaining_ok=True)
    print("✅ Download complete:", args.out_dir)

if __name__ == "__main__":
    main()
