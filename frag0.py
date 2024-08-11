import numpy as np
import matplotlib.pyplot as plt
from google.colab import files
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
import svgwrite

# Upload CSV files
uploaded = files.upload()

# Function to read CSV files
def read_csv(csv_path):
    np_path_XYs = np.genfromtxt(csv_path, delimiter=',')
    path_XYs = []
    for i in np.unique(np_path_XYs[:, 0]):
        npXYs = np_path_XYs[np_path_XYs[:, 0] == i][:, 1:]
        XYs = []
        for j in np.unique(npXYs[:, 0]):
            XY = npXYs[npXYs[:, 0] == j][:, 1:]
            XYs.append(XY)
        path_XYs.append(XYs)
    return path_XYs

# Function to plot the curves
def plot(paths_XYs, title='Plot'):
    fig, ax = plt.subplots(tight_layout=True, figsize=(8, 8))
    colours = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    for i, XYs in enumerate(paths_XYs):
        c = colours[i % len(colours)]
        for XY in XYs:
            ax.plot(XY[:, 0], XY[:, 1], c=c, linewidth=2)
    ax.set_aspect('equal')
    plt.title(title)
    plt.show()

# Load and plot the training data
paths_XYs_frag0 = read_csv('/content/frag0.csv')
plot(paths_XYs_frag0, title='Distorted Image - frag0')

# Load and plot the distortion-free training data
paths_XYs_free_frag0 = read_csv('/content/frag01_sol.csv')
plot(paths_XYs_free_frag0, title='Distortion-Free Image - frag0')

# Function to prepare training data
def prepare_training_data(distorted_paths, free_paths):
    X, y = [], []
    for d_path, f_path in zip(distorted_paths, free_paths):
        for d_XY, f_XY in zip(d_path, f_path):
            if len(d_XY) == len(f_XY):
                X.extend(d_XY)
                y.extend(f_XY)
            else:
                print(f"Warning: Length mismatch between distorted and free paths")
    return np.array(X), np.array(y)

# Prepare the training data
X_frag0, y_frag0 = prepare_training_data(paths_XYs_frag0, paths_XYs_free_frag0)

# Ensure that X_frag0 and y_frag0 have the same number of samples
if len(X_frag0) != len(y_frag0):
    min_len = min(len(X_frag0), len(y_frag0))
    X_frag0 = X_frag0[:min_len]
    y_frag0 = y_frag0[:min_len]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_frag0, y_frag0, test_size=0.2, random_state=42)

# Train a polynomial regression model with L2 regularization
degree = 3
model = make_pipeline(PolynomialFeatures(degree), Ridge(alpha=0.1))
model.fit(X_train, y_train)

# Output the model's performance
print("Model training completed.")
print("Model score on test data: ", model.score(X_test, y_test))

# Function to apply the model to remove distortion
def remove_distortion(paths_XYs, model):
    new_paths = []
    for path in paths_XYs:
        new_path = []
        for XY in path:
            corrected_XY = model.predict(XY)
            new_path.append(corrected_XY)
        new_paths.append(new_path)
    return new_paths

# Apply the model to the distorted data
corrected_paths_frag0 = remove_distortion(paths_XYs_frag0, model)
plot(corrected_paths_frag0, title='Corrected Image - frag0')

# Function to compute cubic bezier control points
def cubic_bezier_control_points(points):
    p = np.array(points)
    n = len(p) - 1
    a, b, c, r = [0] * (n+1), [0] * (n+1), [0] * (n+1), [0] * (n+1)

    b[0] = 2
    c[0] = 1
    r[0] = p[0] + 2 * p[1]

    for i in range(1, n-1):
        a[i] = 1
        b[i] = 4
        c[i] = 1
        r[i] = 4 * p[i] + 2 * p[i+1]

    a[n-1] = 2
    b[n-1] = 7
    r[n-1] = 8 * p[n-1] + p[n]

    b[n] = 2
    r[n] = p[n]

    # Solve for control points
    for i in range(1, n+1):
        m = a[i] / b[i-1]
        b[i] -= m * c[i-1]
        r[i] -= m * r[i-1]

    control_points_1 = [0] * (n+1)
    control_points_2 = [0] * n

    control_points_1[n] = r[n] / b[n]
    for i in range(n-1, -1, -1):
        control_points_1[i] = (r[i] - c[i] * control_points_1[i+1]) / b[i]

    for i in range(n):
        control_points_2[i] = 2 * p[i+1] - control_points_1[i+1]

    return control_points_1, control_points_2

# Function to convert paths to cubic bezier curves
def paths_to_bezier(paths):
    bezier_paths = []
    for path in paths:
        bezier_path = []
        for points in path:
            if len(points) > 1:
                p1, p2 = cubic_bezier_control_points(points)
                bezier_path.append((points, p1, p2))
        bezier_paths.append(bezier_path)
    return bezier_paths

# Convert corrected paths to cubic bezier curves
bezier_paths_frag0 = paths_to_bezier(corrected_paths_frag0)

# Function to generate SVG
def generate_svg(bezier_paths, filename='output.svg'):
    dwg = svgwrite.Drawing(filename, profile='tiny')
    for bezier_path in bezier_paths:
        for points, p1, p2 in bezier_path:
            path_data = f'M {points[0][0]},{points[0][1]}'
            for i in range(1, len(points)):
                path_data += f' C {p1[i-1][0]},{p1[i-1][1]} {p2[i-1][0]},{p2[i-1][1]} {points[i][0]},{points[i][1]}'
            dwg.add(dwg.path(d=path_data, stroke='black', fill='none', stroke_width=2))
    dwg.save()

# Generate the SVG file
generate_svg(bezier_paths_frag0, filename='frag0_corrected.svg')

# Download the generated SVG file
files.download('frag0_corrected.svg')
