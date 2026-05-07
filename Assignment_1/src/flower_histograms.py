# Import libraries 
import os 
import cv2
import csv
import argparse

# File loader function with the optional argument to choose a target image,
# defaults to image_0107.jpg
def file_loader():
    parser = argparse.ArgumentParser(description="Compare flower histograms")
    parser.add_argument("--input", "-i",
                        required=False,
                        default="image_0107.jpg",
                        help="Filename of the flower to compare to")

    args = parser.parse_args()
    
    return args

# Calculate the histogram value of chosen image and
# normalize it using min-max normalization:
def load_data(filepath):
    image = cv2.imread(filepath)
    hist = cv2.calcHist([image],
                                [0, 1, 2],
                                None,
                                [256, 256, 256],
                                [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist, 0, 1.0, cv2.NORM_MINMAX)     
    return hist                      

def main():
    args = file_loader()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    flowers_dir = os.path.join(script_dir, "../data")
    out_dir = os.path.join(script_dir, "../output")
    os.makedirs("../output", exist_ok=True)

    #Load flower
    filepath_flower = os.path.join(flowers_dir, args.input)
    print(f"Target flower: {args.input}")
    hist_flower = load_data(filepath_flower)

    #Loop through all other flowers in the flowers folder
    results = []
    for filename in os.listdir(flowers_dir):
        # Skip the comparison with chosen flower
        if filename == args.input:
            continue

        filepath = os.path.join(flowers_dir, filename)
        hist = load_data(filepath)

     # Compare to the target histogram using Chi-Square
        score = round(cv2.compareHist(hist_flower, hist, cv2.HISTCMP_CHISQR), 2)
        results.append((filename, score))

    # Sort by score and print top 5
    results.sort(key=lambda x: x[1])
    for filename, score in results[:5]:
        print(f"{filename}: {score:.4f}")

    #save top 5 to CSV

    input_stem = os.path.splitext(args.input)[0]
    out_filepath = os.path.join(out_dir, f"{input_stem}_histogram_comparison_results.csv")

    with open(out_filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Filename", "Distance"])
        for filename, score in results[:5]:
            writer.writerow([filename, round(score, 4)])

    print(f"Results saved to {out_filepath}")

if __name__ == "__main__":
    main()