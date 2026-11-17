"""
setup_vessel_assets.py
Ensures local high-resolution maritime vessel imagery is present in assets/vessels/
for guaranteed offline/zero-403 rendering.
"""

import os
import urllib.request
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets", "vessels")
os.makedirs(ASSETS_DIR, exist_ok=True)

IMAGE_SOURCES = {
    "tanker.jpg": "https://images.unsplash.com/photo-1544620347-c4fd4a3d5957?auto=format&fit=crop&w=800&q=80",
    "bulk_carrier.jpg": "https://images.unsplash.com/photo-1518241353330-0f7941c2d9b5?auto=format&fit=crop&w=800&q=80",
    "container.jpg": "https://images.unsplash.com/photo-1586528116311-ad8dd3c8310d?auto=format&fit=crop&w=800&q=80",
    "supply_vessel.jpg": "https://images.unsplash.com/photo-1509316975850-ff9c5deb0cd9?auto=format&fit=crop&w=800&q=80",
    "commercial_default.jpg": "https://images.unsplash.com/photo-1494412574643-ff11b0a5c1c3?auto=format&fit=crop&w=800&q=80"
}


def create_pro_vector_profile(filename: str, ship_type: str, ship_name: str, color: str = "#00ADB5"):
    """Creates a high-resolution dark-themed architectural profile card if network download fails."""
    out_path = os.path.join(ASSETS_DIR, filename)
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor="#0E1117")
    ax.set_facecolor("#161B22")
    
    # Ocean baseline
    ax.axhline(0.25, color="#00ADB5", linestyle="-", linewidth=2.5, alpha=0.85)
    ax.fill_between([-1, 11], -1, 0.25, color="#0D2128", alpha=0.9)
    
    # Hull
    hull_x = [1.0, 1.5, 8.5, 9.2, 8.8, 1.0]
    hull_y = [0.25, 0.15, 0.15, 0.38, 0.42, 0.42]
    ax.add_patch(patches.Polygon(list(zip(hull_x, hull_y)), closed=True, color="#21262D", ec=color, lw=2.5))
    
    # Superstructure & Deck
    ax.add_patch(patches.Rectangle((1.5, 0.42), 1.8, 0.5, color="#30363D", ec="#8B949E", lw=1.5))
    ax.add_patch(patches.Rectangle((1.8, 0.92), 1.2, 0.35, color="#21262D", ec=color, lw=1.5))
    # Radar Mast
    ax.plot([2.4, 2.4], [1.27, 1.65], color="#E6EDF3", lw=2)
    ax.plot([2.1, 2.7], [1.5, 1.5], color="#E6EDF3", lw=1.5)
    
    # Cargo bays / Cranes / Deck equipment
    for cx in [3.8, 5.2, 6.6, 7.8]:
        ax.add_patch(patches.Rectangle((cx, 0.42), 1.0, 0.28, color="#30363D", ec="#484F58", lw=1))
        ax.plot([cx + 0.5, cx + 0.5], [0.70, 0.95], color=color, lw=1.5)
    
    # Waterline wavelets
    for wx in np.linspace(0.5, 9.5, 18):
        ax.plot([wx, wx + 0.3], [0.22, 0.22], color="#00ADB5", alpha=0.5, lw=1.5)
    
    # Text annotations
    ax.text(0.3, 1.75, f"🚢 {ship_name.upper()}", color="#FFFFFF", fontsize=15, fontweight="bold", fontfamily="sans-serif")
    ax.text(0.3, 1.55, f"Classification: {ship_type} | Maritime Forensic Profile", color="#8B949E", fontsize=10, fontfamily="sans-serif")
    ax.text(0.3, 1.38, "Standard AIS Telemetry Track & Hydrodynamic Validation", color="#00ADB5", fontsize=9, fontfamily="sans-serif")
    
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2.0)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print(f"Generated vector profile: {out_path}")


def download_or_create_all():
    headers = {"User-Agent": "MarineSentinel/1.0"}
    for fname, url in IMAGE_SOURCES.items():
        dest = os.path.join(ASSETS_DIR, fname)
        if not os.path.exists(dest):
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=5) as resp:
                    if resp.status == 200:
                        with open(dest, "wb") as f:
                            f.write(resp.read())
                        print(f"Downloaded: {dest}")
                        continue
            except Exception as e:
                print(f"Notice: Download failed for {fname} ({e}), creating vector profile.")
        
        if not os.path.exists(dest):
            base_type = fname.replace(".jpg", "").replace("_", " ").title()
            create_pro_vector_profile(fname, base_type, f"Commercial {base_type}")

    # Specific vessel profiles
    specifics = [
        ("seacor_pride.jpg", "Offshore Supply Vessel", "SEACOR PRIDE", "#FF5252"),
        ("pacific_glory.jpg", "Product Tanker", "PACIFIC GLORY", "#FF0055"),
        ("wakashio.jpg", "Capesize Bulk Carrier", "MV WAKASHIO", "#FF0055"),
        ("cape_flattery.jpg", "Panamax Bulk Carrier", "CAPE FLTERY", "#00ADB5"),
        ("ever_lunar.jpg", "Container Vessel", "EVER LUNAR", "#00ADB5")
    ]
    for s_file, s_type, s_name, s_col in specifics:
        s_dest = os.path.join(ASSETS_DIR, s_file)
        if not os.path.exists(s_dest):
            create_pro_vector_profile(s_file, s_type, s_name, s_col)


if __name__ == "__main__":
    download_or_create_all()
    print("Vessel assets setup complete.")
