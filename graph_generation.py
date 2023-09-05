import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import cv2

# Fixing the seed for reproducibility
np.random.seed(42)

# Directory to save the graphs
directory = os.path.abspath(os.getcwd())+"/content/"
os.makedirs(directory,exist_ok=True)
os.makedirs(directory+"graphs/initial/",exist_ok=True)
os.makedirs(directory+"graphs/processed/",exist_ok=True)

# The arrays to hold the images and the labels
X = []
Y = []

# The number of graphs to generate
num_graphs = 1000

# The range of x values for the graphs
x = np.linspace(-10, 10, 400)

for i in range(num_graphs):
    np.random.seed(29+i)
    # Generate a random degree between 0 and 5
    degree = np.random.randint(6)

    # Generate random coefficients for the polynomial
    coeffs = np.random.uniform(-10, 10, degree + 1)

    # Compute the y values for the polynomial
    y = np.polyval(coeffs, x)

    # Plot the graph
    plt.figure()
    plt.plot(x, y)
    plt.title(f"Polynomial of degree {degree}")
    plt.grid(True)
    plt.savefig(f"{directory}graphs/initial/graph_{i}.png")
    plt.close()

    # Read the saved image
    img = cv2.imread(f"{directory}graphs/initial/graph_{i}.png", cv2.IMREAD_GRAYSCALE)
    
    # Resize the image to a fixed size (100x100)
    img = cv2.resize(img, (100, 100))

    # Add some noise to the image
    noise = np.random.normal(0, 10, img.shape)
    img = img + noise

    # Normalize the image to the range [0, 1]
    img = img / 255.0
    plt.figure()
    plt.imshow(img,cmap="gray")
    plt.savefig(f"{directory}graphs/processed/graph_{i}.png")
    plt.close()
    # Append the image and the degree to the X and Y arrays
    X.append(img)
    Y.append(degree)
X = [arr.tolist() for arr in X]
dataset = pd.DataFrame(data={'Graph': X, 'Degree': Y})
dataset.to_pickle(f"{directory}dataset.pkl")