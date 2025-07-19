import os
import random
from PIL import Image, ImageDraw

SHAPES = ['circle', 'square', 'triangle']
COLORS = ['red', 'green', 'blue']
COLOR_RGB = {
    'red': (255, 0, 0),
    'green': (0, 255, 0),
    'blue': (0, 0, 255)
}
IMG_SIZE = 64
NUM_SAMPLES = 300

def draw_shape(shape, color):
    img = Image.new('RGB', (IMG_SIZE, IMG_SIZE), 'white')
    draw = ImageDraw.Draw(img)
    if shape == 'circle':
        draw.ellipse((16, 16, 48, 48), fill=COLOR_RGB[color])
    elif shape == 'square':
        draw.rectangle((16, 16, 48, 48), fill=COLOR_RGB[color])
    elif shape == 'triangle':
        draw.polygon([(32, 10), (10, 54), (54, 54)], fill=COLOR_RGB[color])
    return img

def generate_dataset(folder, count):
    os.makedirs(folder, exist_ok=True)
    for i in range(count):
        shape = random.choice(SHAPES)
        color = random.choice(COLORS)
        img = draw_shape(shape, color)
        img.save(f"{folder}/{shape}_{color}_{i}.png")

if __name__ == "__main__":
    generate_dataset("data/toy_dataset/train", 300)
    generate_dataset("data/toy_dataset/test", 60)
    print("✅ Dataset created.")