"""Shared grid validation and lossless black/white maps."""
import csv
from pathlib import Path
import numpy as np
from PIL import Image

DATA = Path(__file__).resolve().parent / 'data/land_water_gt_2deg.csv'

def load_grid(path=DATA):
    with open(path, newline='') as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ['latitude', 'longitude', 'label']:
            raise ValueError('Expected latitude,longitude,label columns')
        rows = sorted((int(r['latitude']), int(r['longitude']), r['label']) for r in reader)
    grid = {(a, b) for a in range(-89, 90, 2) for b in range(-179, 180, 2)}
    if len(rows) != 16200 or {(a, b) for a, b, _ in rows} != grid:
        raise ValueError('Expected the complete, unique 16,200-point 2° grid')
    if any(label not in ('Land', 'Water') for _, _, label in rows):
        raise ValueError('Labels must be Land or Water')
    return rows, np.array([label == 'Land' for _, _, label in rows])

def map_image(predictions):
    pixels = np.zeros((16200, 4), dtype=np.uint8)
    pixels[:, :3] = (np.asarray(predictions) == 1)[:, None] * 255
    pixels[:, 3] = (np.asarray(predictions) >= 0) * 255
    return Image.fromarray(pixels.reshape(90, 180, 4)[::-1])
