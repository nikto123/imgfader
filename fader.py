import os
import argparse
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm


def parse_list(path):
    print(f"[INFO] Reading list from: {path}")
    entries = []
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            name, delay = parts
            entries.append((name, float(delay)))
    print(f"[INFO] Loaded {len(entries)} timeline entries")
    return entries


def normalize_timeline(entries, total_duration):
    total_units = sum(delay for _, delay in entries)
    timeline = []
    current_time = 0.0
    for name, delay in entries:
        timeline.append((name, current_time))
        current_time += (delay / total_units) * total_duration
    return timeline


def crossfade(img1_path, img2_path, alpha):
    img1 = Image.open(img1_path).convert("RGB")
    img2 = Image.open(img2_path).convert("RGB")

    arr1 = np.array(img1).astype(np.float32)
    arr2 = np.array(img2).astype(np.float32)
    blended = (arr1 * (1 - alpha) + arr2 * alpha).astype(np.uint8)

    return Image.fromarray(blended)


def generate_frames(timeline, img_dir, frame_count, output_dir, fps, write_video_flag, video_path):
    os.makedirs(output_dir, exist_ok=True)
    times = [t for _, t in timeline]
    names = [n for n, _ in timeline]

    print(f"[INFO] Generating {frame_count} frames at {fps} fps")

    start_time = times[0]
    end_time = times[-1]
    step = (end_time - start_time) / frame_count

    if write_video_flag:
        sample_img = Image.open(os.path.join(img_dir, names[0])).convert("RGB")
        w, h = sample_img.size
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        print(f"[INFO] Writing video to: {video_path}")
    else:
        out = None

    for i in tqdm(range(frame_count), desc="Generating frames"):
        t = start_time + i * step

        for j in range(len(times) - 1):
            if times[j] <= t < times[j + 1]:
                t0, t1 = times[j], times[j + 1]
                name0, name1 = names[j], names[j + 1]
                alpha = (t - t0) / (t1 - t0) if t1 != t0 else 0.0

                path0 = os.path.join(img_dir, name0)
                path1 = os.path.join(img_dir, name1)
                blended = crossfade(path0, path1, alpha)

                if out:
                    out.write(cv2.cvtColor(np.array(blended), cv2.COLOR_RGB2BGR))
                else:
                    out_path = os.path.join(output_dir, f"{i+1:04d}.jpg")
                    blended.save(out_path)
                break
        else:
            last_img = Image.open(os.path.join(img_dir, names[-1])).convert("RGB")
            if out:
                out.write(cv2.cvtColor(np.array(last_img), cv2.COLOR_RGB2BGR))
            else:
                out_path = os.path.join(output_dir, f"{i+1:04d}.jpg")
                last_img.save(out_path)

    if out:
        out.release()
        print("[INFO] Video finished and closed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="./input", help="Input directory with images and list.txt")
    parser.add_argument("--output", default="./output", help="Output directory for generated frames")
    parser.add_argument("--fps", type=int, default=60, help="Frames per second")
    parser.add_argument("--duration", type=float, default=5.0, help="Total duration of the clip in seconds")
    parser.add_argument("--video", action="store_true", help="Generate video instead of separate frames")
    parser.add_argument("--video_path", default="newOut.mp4", help="Output video path")
    args = parser.parse_args()

    list_path = os.path.join(args.input, "list.txt")
    if not os.path.exists(list_path):
        print(f"[ERROR] list.txt not found in {args.input}")
        return

    entries = parse_list(list_path)
    timeline = normalize_timeline(entries, args.duration)
    frame_count = int(args.fps * args.duration)

    generate_frames(timeline, args.input, frame_count, args.output, args.fps, args.video, args.video_path)


if __name__ == "__main__":
    main()

