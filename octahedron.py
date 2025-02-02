import numpy as np
import pyvista as pv
import random
import pickle
import os

# CONSTANTS
THOUSAND = 1000
MILLION = 1000000

# Number of iterations for fractal generation
ITERATIONS = 200 * MILLION
FILE_NAME = "fractal_data"

# Octahedron center and radius
CENTER = [0.0, 0.0, 0.0]  # Centered at origin
RADIUS = 10.0  # Size of octahedron

# Visualization settings
RESOLUTION = (600, 600)  # Image resolution
N_FRAMES = 120  # Number of frames for animation
ZOOM = 2.0  # Camera zoom factor

PKL_FILE = f"{FILE_NAME}.pkl"
GIF_FILE = f"{FILE_NAME}.gif"


def generate_octahedron_points(radius, center):
    """
    Generates the vertices of an octahedron centered at the given coordinates with the given radius.
    """
    points = [
        [center[0] + radius, center[1], center[2]],
        [center[0] - radius, center[1], center[2]],
        [center[0], center[1] + radius, center[2]],
        [center[0], center[1] - radius, center[2]],
        [center[0], center[1], center[2] + radius],
        [center[0], center[1], center[2] - radius],
    ]
    return np.array(points)


def generate_fractal_octahedron(center, radius, iterations):
    """
    Generates a 3D fractal using the octahedron's vertices as attractors.
    """
    points = generate_octahedron_points(radius, center)
    current_point = np.array(center)
    fractal_points = []
    colors = []

    for _ in range(iterations):
        selected_vertex = random.randint(0, len(points) - 1)
        random_vertex = points[selected_vertex]
        current_point = (current_point + random_vertex) / 2
        fractal_points.append(current_point)
        colors.append(selected_vertex)  # Store vertex index for coloring

    return np.array(fractal_points), np.array(colors)


def generate_edges(points):
    """
    Generates the edges of the octahedron for visualization.
    """
    edges = [
        (0, 1), (0, 2), (0, 4),
        (1, 3), (1, 5), (2, 3),
        (2, 4), (3, 5), (4, 5)
    ]
    edge_points = []
    for edge in edges:
        edge_points.append(points[edge[0]])
        edge_points.append(points[edge[1]])
    return np.array(edge_points)


def create_rotation_gif(mesh, lines):
    """
    Generates a rotating GIF of the fractal structure.
    """
    plotter = pv.Plotter(window_size=RESOLUTION, off_screen=True)
    plotter.enable_eye_dome_lighting()
    plotter.add_mesh(mesh, scalars="colors", rgb=True, point_size=2)
    plotter.add_mesh(lines, color="white", line_width=1.5)
    plotter.open_gif(GIF_FILE)

    for i in range(N_FRAMES):
        angle = i * (360 / N_FRAMES)
        plotter.camera_position = [
            (20 * ZOOM * np.cos(np.radians(angle)),
             20 * ZOOM * np.sin(np.radians(angle)),
             10 * ZOOM),
            (0, 0, 0),  # Camera target
            (0, 0, 1)  # Up vector
        ]
        plotter.write_frame()
    
    plotter.close()
    print(f"\033[92mGIF created and saved as '{GIF_FILE}'\033[0m")


def display_model(mesh, lines):
    """
    Displays the 3D model interactively.
    """
    plotter = pv.Plotter()
    plotter.enable_eye_dome_lighting()
    plotter.add_mesh(mesh, scalars="colors", rgb=True, point_size=2)
    plotter.add_mesh(lines, color="white", line_width=1.5)
    plotter.show()


# User input for execution mode
choice = input("""
Select an option: 
1 - Generate new fractal
2 - Load from file
3 - Generate GIF from file

Your choice: """).strip()

if choice == "1" or not os.path.exists(PKL_FILE):
    print("Generating new fractal data...")
    fractal_points, fractal_colors = generate_fractal_octahedron(CENTER, RADIUS, ITERATIONS)
    
    with open(PKL_FILE, "wb") as f:
        pickle.dump((fractal_points, fractal_colors), f)
    print(f"\033[94mData saved to {PKL_FILE}\033[0m")

elif choice == "2" and os.path.exists(PKL_FILE):
    print(f"Loading data from {PKL_FILE}...")
    with open(PKL_FILE, "rb") as f:
        fractal_points, fractal_colors = pickle.load(f)
    
    color_map = np.array([
        [255, 0, 0], [0, 255, 0], [0, 0, 255],
        [255, 255, 0], [255, 0, 255], [0, 255, 255]
    ], dtype=np.uint8)
    colors = color_map[fractal_colors]
    
    mesh = pv.PolyData(fractal_points)
    mesh["colors"] = colors
    
    edges = generate_edges(generate_octahedron_points(RADIUS, CENTER))
    lines = pv.PolyData()
    lines.points = edges
    lines.lines = np.hstack([[2, i, i + 1] for i in range(0, len(edges), 2)])
    
    display_model(mesh, lines)

elif choice == "3" and os.path.exists(PKL_FILE):
    print(f"Loading data from {PKL_FILE} for GIF generation...")
    with open(PKL_FILE, "rb") as f:
        fractal_points, fractal_colors = pickle.load(f)
    
    color_map = np.array([
        [255, 0, 0], [0, 255, 0], [0, 0, 255],
        [255, 255, 0], [255, 0, 255], [0, 255, 255]
    ], dtype=np.uint8)
    colors = color_map[fractal_colors]
    
    mesh = pv.PolyData(fractal_points)
    mesh["colors"] = colors
    
    edges = generate_edges(generate_octahedron_points(RADIUS, CENTER))
    lines = pv.PolyData()
    lines.points = edges
    lines.lines = np.hstack([[2, i, i + 1] for i in range(0, len(edges), 2)])
    
    create_rotation_gif(mesh, lines)

else:
    print("Invalid choice. Exiting.")
