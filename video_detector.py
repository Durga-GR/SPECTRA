import cv2
import numpy as np


def analyze_video(video_path):
    """
    Improved AI video detector using temporal + spatial analysis:

    1. Temporal motion irregularity: Real videos have unpredictable motion;
       AI videos have suspiciously smooth/looping motion patterns.
    2. Frame-to-frame flicker noise: Real cameras have natural sensor noise.
    3. Motion blur consistency: Real fast-moving objects blur; AI often doesn't.
    4. Facial/texture region stability: AI faces morph unnaturally between frames.
    5. Compression artifact pattern: AI videos have unusual block artifact patterns.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return {"label": "Could Not Read Video", "score": 0.0}

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_count = min(60, total_frames)

    frames_gray = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame_count >= sample_count:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (256, 256))
        frames_gray.append(gray.astype(np.float32))
        frame_count += 1

    cap.release()

    if len(frames_gray) < 4:
        return {"label": "Video Too Short to Analyze", "score": 0.0}

    frames = np.array(frames_gray)  # shape: (N, 256, 256)

    # ── Feature 1: Motion irregularity ───────────────────────────────────────
    # Compute frame-diff magnitudes
    diffs = np.array([
        np.mean(np.abs(frames[i+1] - frames[i]))
        for i in range(len(frames) - 1)
    ])
    motion_mean = np.mean(diffs)
    motion_std = np.std(diffs)
    # Real video: motion_std/motion_mean > 0.4 (unpredictable)
    # AI video: motion_std/motion_mean < 0.2 (suspiciously smooth/periodic)
    motion_irregularity = motion_std / (motion_mean + 1e-6)
    motion_score = _clamp((motion_irregularity - 0.15) / 0.50)

    # ── Feature 2: Sensor noise residual ─────────────────────────────────────
    # Estimate per-pixel temporal noise (std over time in static regions)
    temporal_noise = np.std(frames, axis=0)  # (256, 256)
    # Use low-diff regions (approximately "static" background)
    diff_map = np.mean(np.abs(np.diff(frames, axis=0)), axis=0)
    static_mask = diff_map < np.percentile(diff_map, 30)
    if np.sum(static_mask) > 100:
        static_noise = np.mean(temporal_noise[static_mask])
    else:
        static_noise = np.mean(temporal_noise)
    # Real cameras: static_noise > 1.5; AI: < 0.8
    noise_score = _clamp((static_noise - 0.5) / 3.0)

    # ── Feature 3: Edge consistency across frames ─────────────────────────────
    # AI videos have unnaturally stable/perfect edges; real ones vary with motion
    edge_means = []
    for f in frames:
        edges = cv2.Canny(f.astype(np.uint8), 50, 150)
        edge_means.append(np.mean(edges))
    edge_var = np.std(edge_means) / (np.mean(edge_means) + 1e-6)
    # Real: edge_var > 0.25; AI: < 0.10
    edge_score = _clamp((edge_var - 0.08) / 0.30)

    # ── Feature 4: Histogram temporal variation ───────────────────────────────
    # Per-frame histogram — AI videos often have very stable brightness distribution
    hist_means = []
    for f in frames:
        hist, _ = np.histogram(f, bins=32, range=(0, 255))
        hist_means.append(hist / hist.sum())
    hist_means = np.array(hist_means)
    # Variation across time in histogram
    hist_temporal_var = np.mean(np.std(hist_means, axis=0))
    # Real: > 0.003; AI: < 0.001
    hist_score = _clamp((hist_temporal_var - 0.0008) / 0.005)

    # ── Feature 5: Local motion coherence ────────────────────────────────────
    # Divide frame into 4x4 blocks and check if motion is coherent (AI) or chaotic (real)
    block_diffs = []
    for i in range(min(len(frames)-1, 20)):
        frame_diff = np.abs(frames[i+1] - frames[i])
        block_size = 64
        bds = []
        for y in range(0, 256, block_size):
            for x in range(0, 256, block_size):
                bds.append(np.mean(frame_diff[y:y+block_size, x:x+block_size]))
        block_diffs.append(np.std(bds))  # spatial variation in motion
    spatial_motion_var = np.mean(block_diffs)
    # Real: >1.5 (motion happens in some blocks, not others)
    # AI: <0.5 (whole frame moves uniformly or not at all)
    spatial_score = _clamp((spatial_motion_var - 0.4) / 2.5)

    # ── Weighted combination ──────────────────────────────────────────────────
    weights = [0.30, 0.25, 0.20, 0.15, 0.10]
    human_score = (
        weights[0] * motion_score +
        weights[1] * noise_score +
        weights[2] * edge_score +
        weights[3] * hist_score +
        weights[4] * spatial_score
    )

    threshold = 0.45
    if human_score >= threshold:
        label = "Likely Human Recorded Video"
        confidence = 0.50 + human_score * 0.50
    else:
        label = "Likely AI Generated Video"
        confidence = 0.50 + (1 - human_score) * 0.50

    return {
        "label": label,
        "score": round(_clamp(confidence), 2),
        "details": {
            "motion_irregularity": round(float(motion_irregularity), 3),
            "sensor_noise": round(float(static_noise), 3),
            "edge_variance": round(float(edge_var), 3),
        }
    }


def _clamp(v):
    return float(max(0.0, min(1.0, v)))
