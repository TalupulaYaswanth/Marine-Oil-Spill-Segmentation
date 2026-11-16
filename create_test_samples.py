"""
create_test_samples.py
Generates a curated suite of realistic SAR test images for testing the
Oil Spill Segmentation and Look-Alike Classification Engine:
  1. Real Heavy Oil Spill (High damping contrast > 8 dB, sharp boundary, vessel nearby)
  2. Low-Wind Zone Look-Alike (Diffuse calm water, low damping ~3.5 dB)
  3. Biogenic Surfactant / Algal Film (Fibrous natural filament, moderate damping)
  4. Clean Open Ocean with Vessel Targets (Uniform sea clutter, bright metallic scatterers)
  5. MV Wakashio Disaster Scene (Large marine casualty slick)
"""

import os
import sys
import numpy as np
from PIL import Image

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

os.environ["MPLCONFIGDIR"] = os.path.join(os.environ.get("TEMP", "/tmp"), "mpl_cache")

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "test_samples")
os.makedirs(OUTPUT_DIR, exist_ok=True)

HEIGHT, WIDTH = 512, 512

def save_image(array_norm: np.ndarray, filename_base: str):
    """Saves normalized [0, 1] array as 8-bit PNG and 16-bit GeoTIFF-compatible image."""
    img_8bit = np.clip(array_norm * 255.0, 0, 255).astype(np.uint8)
    png_path = os.path.join(OUTPUT_DIR, f"{filename_base}.png")
    Image.fromarray(img_8bit).save(png_path)
    
    # Also save as numpy array
    npy_path = os.path.join(OUTPUT_DIR, f"{filename_base}.npy")
    np.save(npy_path, array_norm.astype(np.float32))
    print(f"Saved: {png_path} and {npy_path}")

def generate_samples():
    np.random.seed(42)

    # ---------------------------------------------------------
    # 1. Real Heavy Oil Spill
    # ---------------------------------------------------------
    sea1 = np.random.gamma(shape=3.0, scale=0.03, size=(HEIGHT, WIDTH)).astype(np.float32)
    y, x = np.ogrid[:HEIGHT, :WIDTH]
    # Elliptical slick with tail
    slick1 = (((x - 240) / 70) ** 2 + ((y - 250) / 32) ** 2) <= 1.0
    tail1 = (((x - 330) / 50) ** 2 + ((y - 270) / 16) ** 2) <= 1.0
    sea1[slick1 | tail1] *= 0.10  # 10 dB damping
    # Metallic ship at origin of slick
    sea1[248:253, 168:173] = 3.5
    db1 = 10.0 * np.log10(sea1 + 1e-7)
    p1, p99 = np.percentile(db1, (1.0, 99.0))
    norm1 = np.clip((db1 - p1) / max(p99 - p1, 1e-5), 0.0, 1.0)
    save_image(norm1, "sample_1_real_oil_spill")

    # ---------------------------------------------------------
    # 2. Low-Wind Zone Look-Alike
    # ---------------------------------------------------------
    np.random.seed(101)
    sea2 = np.random.gamma(shape=3.0, scale=0.03, size=(HEIGHT, WIDTH)).astype(np.float32)
    # Diffuse, large calm zone with soft Gaussian edges
    from scipy.ndimage import gaussian_filter
    calm_field = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    calm_field[150:380, 120:400] = 1.0
    calm_smooth = gaussian_filter(calm_field, sigma=45.0)
    sea2 = sea2 * (1.0 - 0.55 * calm_smooth)  # Moderate damping ~3-4 dB with soft gradient
    db2 = 10.0 * np.log10(sea2 + 1e-7)
    p1, p99 = np.percentile(db2, (1.0, 99.0))
    norm2 = np.clip((db2 - p1) / max(p99 - p1, 1e-5), 0.0, 1.0)
    save_image(norm2, "sample_2_low_wind_lookalike")

    # ---------------------------------------------------------
    # 3. Biogenic Surfactant / Algal Bloom Look-Alike
    # ---------------------------------------------------------
    np.random.seed(202)
    sea3 = np.random.gamma(shape=3.0, scale=0.03, size=(HEIGHT, WIDTH)).astype(np.float32)
    # Sinusoidal fibrous filaments characteristic of natural organic slicks
    xx, yy = np.meshgrid(np.linspace(0, 4 * np.pi, WIDTH), np.linspace(0, 4 * np.pi, HEIGHT))
    filaments = np.sin(xx + np.sin(yy)) * np.cos(yy * 0.8)
    filament_mask = (filaments > 0.65)
    sea3[filament_mask] *= 0.35  # ~4.5 dB damping
    db3 = 10.0 * np.log10(sea3 + 1e-7)
    p1, p99 = np.percentile(db3, (1.0, 99.0))
    norm3 = np.clip((db3 - p1) / max(p99 - p1, 1e-5), 0.0, 1.0)
    save_image(norm3, "sample_3_biogenic_algae_lookalike")

    # ---------------------------------------------------------
    # 4. Clean Open Ocean with Ships
    # ---------------------------------------------------------
    np.random.seed(303)
    sea4 = np.random.gamma(shape=3.2, scale=0.03, size=(HEIGHT, WIDTH)).astype(np.float32)
    # Multiple ships sailing without any slick
    vessels = [(120, 200), (380, 150), (250, 410), (440, 320)]
    for vx, vy in vessels:
        sea4[vy-2:vy+3, vx-2:vx+3] = 3.2
    db4 = 10.0 * np.log10(sea4 + 1e-7)
    p1, p99 = np.percentile(db4, (1.0, 99.0))
    norm4 = np.clip((db4 - p1) / max(p99 - p1, 1e-5), 0.0, 1.0)
    save_image(norm4, "sample_4_clean_ocean_with_ships")

    # ---------------------------------------------------------
    # 5. MV Wakashio Disaster Scene
    # ---------------------------------------------------------
    np.random.seed(404)
    sea5 = np.random.gamma(shape=2.9, scale=0.03, size=(HEIGHT, WIDTH)).astype(np.float32)
    # Large coastal slick wrapping around reef
    reef_x, reef_y = 180, 260
    dist_reef = np.sqrt((x - reef_x) ** 2 + (y - reef_y) ** 2)
    slick_wak = (dist_reef < 95) & (x > reef_x - 30) & (y > reef_y - 80)
    sea5[slick_wak] *= 0.09  # 11 dB heavy fuel oil damping
    sea5[reef_y-3:reef_y+4, reef_x-3:reef_x+4] = 4.0  # Grounded bulk carrier
    db5 = 10.0 * np.log10(sea5 + 1e-7)
    p1, p99 = np.percentile(db5, (1.0, 99.0))
    norm5 = np.clip((db5 - p1) / max(p99 - p1, 1e-5), 0.0, 1.0)
    save_image(norm5, "sample_5_wakashio_disaster")

if __name__ == "__main__":
    generate_samples()
    print("Curated benchmark test samples created successfully in data/test_samples/!")
