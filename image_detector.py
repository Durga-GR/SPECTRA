from PIL import Image
import numpy as np


def analyze_image(image_file):
    """
    Improved AI image detector using multiple heuristics:
    1. Noise pattern analysis (AI images are often too smooth/clean)
    2. Color distribution uniformity
    3. Local texture entropy
    4. High-frequency detail sharpness
    5. Chroma channel uniformity (AI images have unnaturally smooth color)
    """
    image = Image.open(image_file).convert("RGB")
    image = image.resize((512, 512))
    img_array = np.array(image, dtype=np.float32)

    gray = np.mean(img_array, axis=2)

    # ── Feature 1: Noise residual (Laplacian-style high-pass filter) ──────────
    # AI images are smoother → lower noise residual after high-pass filtering
    kernel = np.array([
        [-1, -1, -1],
        [-1,  8, -1],
        [-1, -1, -1]
    ], dtype=np.float32)
    # Manual 2D convolution (no scipy needed)
    from PIL import ImageFilter
    gray_pil = Image.fromarray(gray.astype(np.uint8))
    edges_pil = gray_pil.filter(ImageFilter.FIND_EDGES)
    edge_array = np.array(edges_pil, dtype=np.float32)
    noise_level = np.mean(edge_array)  # Higher = more natural micro-detail

    # ── Feature 2: Local block entropy ───────────────────────────────────────
    # Real photos have varied entropy across regions; AI images are often
    # globally smooth with sudden sharp feature boundaries
    block_stds = []
    step = 64
    for y in range(0, 512 - step, step):
        for x in range(0, 512 - step, step):
            block = gray[y:y+step, x:x+step]
            block_stds.append(np.std(block))
    block_stds = np.array(block_stds)
    entropy_variation = np.std(block_stds)   # Low = suspiciously uniform

    # ── Feature 3: Chroma smoothness ─────────────────────────────────────────
    # AI generators tend to over-smooth color channels
    r_std = np.std(img_array[:, :, 0])
    g_std = np.std(img_array[:, :, 1])
    b_std = np.std(img_array[:, :, 2])
    channel_balance = np.std([r_std, g_std, b_std])  # Low = suspiciously balanced

    # ── Feature 4: Pixel value histogram flatness ────────────────────────────
    # Real images have spiked histograms; AI images are often flatter/smoothed
    hist, _ = np.histogram(gray.flatten(), bins=64, range=(0, 255))
    hist_norm = hist / hist.sum()
    hist_spikiness = np.max(hist_norm) / (np.mean(hist_norm) + 1e-6)
    # Low spikiness → suspiciously flat histogram

    # ── Feature 5: Edge coherence ────────────────────────────────────────────
    # AI images have suspiciously perfect, coherent edges
    edge_flat = edge_array.flatten()
    high_edge_ratio = np.sum(edge_flat > 50) / len(edge_flat)
    # Compute how "clean" edges are: low variance around edge pixels
    edge_pixels = edge_flat[edge_flat > 30]
    edge_coherence = np.std(edge_pixels) if len(edge_pixels) > 100 else 30

    # ── Scoring ───────────────────────────────────────────────────────────────
    # Normalize each feature to a 0–1 human-ness score
    #  noise:           >12 → human,  <5 → AI
    noise_score = _clamp((noise_level - 5) / 12)

    #  entropy_var:     >15 → human, <5 → AI
    entropy_score = _clamp((entropy_variation - 5) / 15)

    #  channel_balance: >3 → human, <1 → AI
    channel_score = _clamp((channel_balance - 1) / 6)

    #  hist_spikiness:  >3 → human, <1.5 → AI
    hist_score = _clamp((hist_spikiness - 1.5) / 4)

    #  edge_coherence:  >20 → human, <10 → AI
    edge_score = _clamp((edge_coherence - 10) / 20)

    # Weighted average
    weights = [0.30, 0.25, 0.20, 0.15, 0.10]
    human_score = (
        weights[0] * noise_score +
        weights[1] * entropy_score +
        weights[2] * channel_score +
        weights[3] * hist_score +
        weights[4] * edge_score
    )

    threshold = 0.45
    if human_score >= threshold:
        label = "Likely Human Generated Image"
        confidence = 0.50 + human_score * 0.50
    else:
        label = "Likely AI Generated Image"
        confidence = 0.50 + (1 - human_score) * 0.50

    return {
        "label": label,
        "score": round(_clamp(confidence), 2),
        "details": {
            "noise_level": round(float(noise_level), 2),
            "entropy_variation": round(float(entropy_variation), 2),
            "channel_balance": round(float(channel_balance), 2),
        }
    }


def _clamp(v):
    return float(max(0.0, min(1.0, v)))
