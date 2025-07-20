import requests, ndjson
import numpy as np
from tqdm import tqdm
from pathlib import Path
import cv2
from scipy.interpolate import interp1d

# -----------------------------
# Step 1: Define categories
# -----------------------------
from categories import getCategories
categories = getCategories()


cat_to_label = {cat: i for i, cat in enumerate(categories)}

base_url = "https://storage.googleapis.com/quickdraw_dataset/full/simplified/"
data_dir = Path("quickdraw_data")
data_dir.mkdir(exist_ok=True)

# -----------------------------
# Step 2: Download if not exists
# -----------------------------
for cat in categories:
    file_path = data_dir / f"{cat}.ndjson"
    if not file_path.exists():
        print(f"⬇️ Downloading {cat}.ndjson...")
        r = requests.get(base_url + f"{cat}.ndjson")
        with open(file_path, 'wb') as f:
            f.write(r.content)

# -----------------------------
# RDP simplification
# -----------------------------
def rdp(points, epsilon=2.0):
    if len(points) < 3:
        return points
    from shapely.geometry import LineString
    line = LineString(points)
    simplified = line.simplify(epsilon)
    return list(simplified.coords)

# -----------------------------
# Resample stroke to 1-pixel spacing
# -----------------------------
def resample_stroke(stroke):
    x, y = stroke
    points = np.array(list(zip(x, y)))
    dists = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))
    total_len = np.sum(dists)
    if total_len == 0:
        return [x, y]
    num_points = max(int(total_len), 1)
    t = np.linspace(0, 1, len(points))
    t_new = np.linspace(0, 1, num_points)
    interp_x = interp1d(t, points[:, 0], kind='linear')(t_new)
    interp_y = interp1d(t, points[:, 1], kind='linear')(t_new)
    return [interp_x.astype(int).tolist(), interp_y.astype(int).tolist()]

# -----------------------------
# Draw strokes with full preprocessing
# -----------------------------
def draw_bitmap(strokes, size=28):
    full_canvas_size = 256
    target_core_size = 24  # resize result before padding
    margin = (size - target_core_size) // 2

    all_points = np.concatenate([np.stack(stroke, axis=-1) for stroke in strokes])
    min_xy = all_points.min(axis=0)
    max_xy = all_points.max(axis=0)
    range_xy = max_xy - min_xy

    if range_xy.max() == 0:
        return np.zeros((size, size), dtype=np.uint8)

    scale = full_canvas_size / range_xy.max()
    img = np.zeros((full_canvas_size, full_canvas_size), dtype=np.uint8)

    for stroke in strokes:
        x = (np.array(stroke[0]) - min_xy[0]) * scale
        y = (np.array(stroke[1]) - min_xy[1]) * scale
        stroke = [x.astype(int).tolist(), y.astype(int).tolist()]
        stroke = resample_stroke(stroke)
        pts = np.array(list(zip(stroke[0], stroke[1])))
        pts = rdp(pts.tolist(), epsilon=2.0)

        for i in range(len(pts) - 1):
            pt1 = tuple(map(int, pts[i]))
            pt2 = tuple(map(int, pts[i + 1]))
            cv2.line(img, pt1, pt2, 255, 1)

    # Resize to 24x24
    resized = cv2.resize(img, (target_core_size, target_core_size), interpolation=cv2.INTER_AREA)

    # Pad to 28x28 with 2-pixel margin
    padded = np.pad(resized, pad_width=margin, mode='constant', constant_values=0)
    padded[padded > 0] = 255
    return padded

# -----------------------------
# Step 4: Parse and preprocess
# -----------------------------
X = []
y = []

samples_per_class = 10000  # Change as needed

for cat in tqdm(categories, desc="Processing categories"):
    file_path = data_dir / f"{cat}.ndjson"
    with open(file_path) as f:
        data = ndjson.load(f)

    for item in data[:samples_per_class]:
        img = draw_bitmap(item["drawing"])
        X.append(img)
        y.append(cat_to_label[cat])

X = np.array(X).astype(np.uint8)
y = np.array(y).astype(np.int32)

print("✅ Shape of X:", X.shape)
print("✅ Shape of y:", y.shape)

# -----------------------------
# Step 5: Save to disk
# -----------------------------
np.save("X.npy", X)
np.save("y.npy", y)

print("✅ Saved X.npy and y.npy")
