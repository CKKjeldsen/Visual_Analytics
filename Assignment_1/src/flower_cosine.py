# Import libraries 
import os
import csv
import argparse
import numpy as np
from numpy.linalg import norm
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
 
# File loader function with the optional argument to choose a target image,
# defaults to image_0107.jpg 
def file_loader():
    parser = argparse.ArgumentParser(description="Compare flower embeddings using VGG16")
    parser.add_argument("--input", "-i",
                        required=False,
                        default="image_0107.jpg",
                        help="Filename of the flower to compare to")
 
    args = parser.parse_args()
 
    return args
 
# Load the pretrained VGG16 model without it's classifier head
def load_model():
    model = VGG16(weights="imagenet",
                  include_top=False,
                  pooling="avg") 
    return model
 
# Load the data and resize it to VGG16's desired size, 224x224
# Convert the image to a numpy array and normalize 
# Run image through model to get embedding, flatten to a 1D vector 
def load_data(filepath, model):
    image = load_img(filepath, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)  
    image = preprocess_input(image)         
    embedding = model.predict(image, verbose=0)
    embedding = embedding.flatten()
    return embedding
 
# Calculate cosine similarity 
def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))
 
 
def main():
    args = file_loader()
 
    script_dir = os.path.dirname(os.path.abspath(__file__))
    flowers_dir = os.path.join(script_dir, "../data")
    out_dir = os.path.join(script_dir, "../output")
    os.makedirs("../output", exist_ok=True)
 
    model = load_model()
 
    # Load target flower embedding
    filepath_flower = os.path.join(flowers_dir, args.input)
    print(f"Target flower image: {args.input}")
    embedding_flower = load_data(filepath_flower, model)
 
    # Loop through all other flowers in the flowers folder and compute
    # cosine similarity 
    results = []
    for filename in os.listdir(flowers_dir):
        if filename == args.input:
            continue
 
        filepath = os.path.join(flowers_dir, filename)
        embedding = load_data(filepath, model)
 
        score = round(cosine_similarity(embedding_flower, embedding), 4)
        results.append((filename, score))
 
    # Save top 5 to CSV
    results.sort(key=lambda x: x[1], reverse=True)

    for filename, score in results[:5]:
        print(f"{filename}: {score:.4f}")
 
  
    input_stem = os.path.splitext(args.input)[0]
    out_filepath = os.path.join(out_dir, f"{input_stem}_cosine_similiarity_comparison_results.csv")

    with open(out_filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Filename", "Cosine Similarity"])
        for filename, score in results[:5]:
            writer.writerow([filename, score])

    print(f"Results saved to {out_filepath}")
 
 
if __name__ == "__main__":
    main()