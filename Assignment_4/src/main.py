"""
Newspaper face detection analysis script

Detects human faces in historical Swiss newspaper pages (JDG, GDL, IMP)
using MTCNN, groups results by decade, and outputs CSVs and plots.
"""

# Import libraries 
import os
import re
import csv
from pathlib import Path
from collections import defaultdict
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from facenet_pytorch import MTCNN


NEWSPAPERS = ["JDG", "GDL", "IMP"]
NEWSPAPERS_ROOT = Path("../data/newspapers")

# Regex to extract the date from filenames 
FILENAME_RE = re.compile(r"[A-Z]+-(\d{4})-\d{2}-\d{2}-.*\.jpg", re.IGNORECASE)

OUTPUT_DIR = Path(__file__).parent.parent / "output"

# Utility functions

# Extract the decade that a year belongs to
def get_decade(year: int) -> int:
    return (year // 10) * 10

# Parse the year from a newspaper image filename.
def extract_year(filename: str) -> int | None:
    match = FILENAME_RE.match(filename)
    if match:
        return int(match.group(1))
    return None

# Face detection function, which runs face detection on a single image and 
# returns the number of faces found. Returns 0 if the image cannot 
# be opened or no faces are detected.

def count_faces(mtcnn: MTCNN, image_path: Path) -> int:
    try:
        img = Image.open(image_path).convert("RGB")

        boxes, _ = mtcnn.detect(img)
        if boxes is None:
            return 0
        return len(boxes)
    except Exception as exc:
        print(f"    Could not process {image_path.name}: {exc}")
        return 0


 # Go through all page images for each newspaper and detect faces, aggregate by decade.
 # Returns a dict keyed by decade
def analyse_newspaper(newspaper: str, mtcnn: MTCNN) -> dict[int, dict]:

    newspaper_dir = NEWSPAPERS_ROOT / newspaper

    # Collect all files
    image_paths = sorted(newspaper_dir.glob("*.jpg"))
    total_images = len(image_paths)

    decade_data: dict[int, dict] = defaultdict(
        lambda: {"total_faces": 0, "pages_total": 0, "pages_with_faces": 0}
    )

    for idx, img_path in enumerate(image_paths, start=1):
        year = extract_year(img_path.name)
        if year is None:
            continue

        decade = get_decade(year)
        n_faces = count_faces(mtcnn, img_path)

        decade_data[decade]["total_faces"] += n_faces
        decade_data[decade]["pages_total"] += 1
        if n_faces > 0:
            decade_data[decade]["pages_with_faces"] += 1

        # Progress indicator every 100 pages to follow along in terminal 
        if idx % 100 == 0 or idx == total_images:
            print(f"    Processed {idx}/{total_images} pages …")

    return dict(decade_data)

# Output csv function

def save_csv(newspaper: str, decade_data: dict[int, dict], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{newspaper}_faces_by_decade.csv"

    fieldnames = [
        "decade",
        "total_faces",
        "pages_total",
        "pages_with_faces",
        "pct_pages_with_faces",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()

        for decade in sorted(decade_data):
            row = decade_data[decade]
            pages_total = row["pages_total"]
            pages_with_faces = row["pages_with_faces"]
            pct = (pages_with_faces / pages_total * 100) if pages_total > 0 else 0.0

            writer.writerow(
                {
                    "decade": decade,
                    "total_faces": row["total_faces"],
                    "pages_total": pages_total,
                    "pages_with_faces": pages_with_faces,
                    "pct_pages_with_faces": round(pct, 2),
                }
            )

    print(f"  CSV saved to {csv_path}")
    return csv_path

# Output plot function 

def save_plot(newspaper: str, decade_data: dict[int, dict], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_path = out_dir / f"{newspaper}_pct_pages_with_faces.png"

    decades = sorted(decade_data)
    pcts = []
    for d in decades:
        row = decade_data[d]
        pct = (
            row["pages_with_faces"] / row["pages_total"] * 100
            if row["pages_total"] > 0
            else 0.0
        )
        pcts.append(pct)

    decade_labels = [str(d) for d in decades]

    fig, ax = plt.subplots(figsize=(max(8, len(decades) * 0.9), 5))
    bars = ax.bar(decade_labels, pcts, color="blue", edgecolor="white", width=0.6)

    for bar, pct in zip(bars, pcts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{pct:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_title(
        f"{newspaper} – Percentage of pages containing ≥1 face per decade",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )
    ax.set_xlabel("Decade", fontsize=11)
    ax.set_ylabel("% pages with faces", fontsize=11)
    ax.set_ylim(0, min(100, max(pcts, default=10) * 1.25 + 5))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    print(f"  Plot saved to {plot_path}")
    return plot_path

# Main function, analyse all images in each newspaper folder

def main() -> None:
    print("=" * 60)
    print("  Newspaper Face Detection Analysis")
    print("=" * 60)

    out_dir = OUTPUT_DIR
    # Initialize MTCNN for face detection
    mtcnn = MTCNN(keep_all=True)

    for newspaper in NEWSPAPERS:
        decade_data = analyse_newspaper(newspaper, mtcnn)

        if not decade_data:
            continue

        save_csv(newspaper, decade_data, out_dir)
        save_plot(newspaper, decade_data, out_dir)
        print(f"[{newspaper}] Done.\n")

    print("\nAll newspapers processed. Results are in:", out_dir.resolve())


if __name__ == "__main__":
    main()
