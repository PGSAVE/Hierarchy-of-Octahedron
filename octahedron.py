import numpy as np
import pyvista as pv
import random
import pickle
import os

# ПАРАМЕТРЫ
thousand = 1000
million = 1000000

# Количество итераций
iterations = 200 * million
file = "fractal_data"

# Центр и радиус октаэдра
center = [0.0, 0.0, 0.0]  # Центрирование в начале координат
radius = 10.0  # Размер октаэдра

resolution = (600, 600)  # Разрешение изображения
n_frames = 120  # Количество кадров в анимации
zoom = 2.0  # Увеличение


# Функция для генерации точек трёхмерного фрактала
# Используем вершины октаэдра для трёхмерной структуры
def generate_octahedron_points(radius, center):
    points = [
        [center[0] + radius, center[1], center[2]],
        [center[0] - radius, center[1], center[2]],
        [center[0], center[1] + radius, center[2]],
        [center[0], center[1] - radius, center[2]],
        [center[0], center[1], center[2] + radius],
        [center[0], center[1], center[2] - radius],
    ]
    return np.array(points)

pkl_file = f"{file}.pkl"
gif_file = f"{file}.gif"

# Функция для генерации точек фрактала "треугольник Серпинского в 3D"
def generate_fractal_octahedron(center, radius, iterations):
    points = generate_octahedron_points(radius, center)
    current_point = np.array(center)
    fractal_points = []
    colors = []

    for _ in range(iterations):
        selected_vertex = random.randint(0, len(points) - 1)
        random_vertex = points[selected_vertex]
        current_point = (current_point + random_vertex) / 2
        fractal_points.append(current_point)
        colors.append(selected_vertex)  # Сохраняем индекс вершины для цвета

    return np.array(fractal_points), np.array(colors)

# Функция для генерации рёбер октаэдра
def generate_edges(points):
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

# Функция для запуска визуализации и вращения
def create_rotation_gif(mesh, lines):
    plotter = pv.Plotter(window_size=resolution, off_screen=True)
    plotter.enable_eye_dome_lighting()
    plotter.add_mesh(mesh, scalars="colors", rgb=True, point_size=2)
    plotter.add_mesh(lines, color="white", line_width=1.5)

    plotter.open_gif(gif_file)
    for i in range(n_frames):
        angle = i * (360 / n_frames)
        # Используем переменную zoom для управления расстоянием
        plotter.camera_position = [
            (20 * zoom * np.cos(np.radians(angle)),
             20 * zoom * np.sin(np.radians(angle)),
             10 * zoom),
            (0, 0, 0),  # Точка, на которую смотрит камера
            (0, 0, 1)  # Вектор "вверх"
        ]
        plotter.write_frame()
    plotter.close()
    print(f"\033[92mGIF создан и сохранён как '{gif_file}'\033[0m")

# Функция для отображения 3D модели
def display_model(mesh, lines):
    plotter = pv.Plotter()
    plotter.enable_eye_dome_lighting()
    plotter.add_mesh(mesh, scalars="colors", rgb=True, point_size=2)
    plotter.add_mesh(lines, color="white", line_width=1.5)
    plotter.show()  # Открывает окно для просмотра модели

# Основная логика программы
choice = input("""
Выберите действие: 
1 - Сгенерировать снова
2 - Загрузить из файла
3 - Сгенерировать GIF из файла: 

Ваш выбор: """).strip()

if choice == "1" or not os.path.exists(pkl_file):
    # Генерация точек фрактала в 3D
    print("Генерация новых данных...")
    fractal_points, fractal_colors = generate_fractal_octahedron(center, radius, iterations)

    # Сохранение данных в pkl файл
    with open(pkl_file, "wb") as f:
        pickle.dump((fractal_points, fractal_colors), f)
    print(f"\033[94mДанные сохранены в {pkl_file}\033[0m")

elif choice == "2" and os.path.exists(pkl_file):
    # Загрузка данных из файла
    print(f"Загрузка данных из {pkl_file}...")
    with open(pkl_file, "rb") as f:
        fractal_points, fractal_colors = pickle.load(f)

    # Преобразование индексов вершин в цвета
    color_map = np.array([
        [255, 0, 0],  # Красный
        [0, 255, 0],  # Зелёный
        [0, 0, 255],  # Синий
        [255, 255, 0],  # Жёлтый
        [255, 0, 255],  # Магента
        [0, 255, 255],  # Циан
    ], dtype=np.uint8)
    colors = color_map[fractal_colors]

    # Создание объекта для визуализации точек
    mesh = pv.PolyData(fractal_points)
    mesh["colors"] = colors

    # Создание рёбер октаэдра
    edges = generate_edges(generate_octahedron_points(radius, center))
    lines = pv.PolyData()
    lines.points = edges
    lines.lines = np.hstack([[2, i, i + 1] for i in range(0, len(edges), 2)])

    # Отображение модели
    display_model(mesh, lines)

elif choice == "3" and os.path.exists(pkl_file):
    # Загрузка данных из файла
    print(f"Загрузка данных из {pkl_file} для генерации GIF...")
    with open(pkl_file, "rb") as f:
        fractal_points, fractal_colors = pickle.load(f)

    # Преобразование индексов вершин в цвета
    color_map = np.array([
        [255, 0, 0],  # Красный
        [0, 255, 0],  # Зелёный
        [0, 0, 255],  # Синий
        [255, 255, 0],  # Жёлтый
        [255, 0, 255],  # Магента
        [0, 255, 255],  # Циан
    ], dtype=np.uint8)
    colors = color_map[fractal_colors]

    # Создание объекта для визуализации точек
    mesh = pv.PolyData(fractal_points)
    mesh["colors"] = colors

    # Создание рёбер октаэдра
    edges = generate_edges(generate_octahedron_points(radius, center))
    lines = pv.PolyData()
    lines.points = edges
    lines.lines = np.hstack([[2, i, i + 1] for i in range(0, len(edges), 2)])

    # Генерация GIF
    create_rotation_gif(mesh, lines)

else:
    print("Некорректный выбор. Завершение программы.")
