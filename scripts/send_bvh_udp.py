"""Send BVH motion data over UDP for testing the online sim pipeline.

Usage:
    python scripts/send_bvh_udp.py --bvh data/hc_mocap/wander.bvh --loop
    python scripts/send_bvh_udp.py --bvh data/hc_mocap/wander.bvh --fps 30 --port 1118
    python scripts/send_bvh_udp.py --bvh data/hc_mocap/wander.bvh --downsample 2
"""

from __future__ import annotations

import argparse
import socket
import time


def _read_motion_lines(bvh_path: str) -> tuple[list[str], float]:
    """Read the MOTION section of a BVH file.

    Returns (list of data lines, frame_time in seconds).
    """
    lines: list[str] = []
    frame_time = 1.0 / 30.0
    in_motion = False
    past_header = False

    with open(bvh_path, "r") as f:
        for line in f:
            stripped = line.strip()
            if stripped == "MOTION":
                in_motion = True
                continue
            if not in_motion:
                continue
            if stripped.startswith("Frames:"):
                continue
            if stripped.startswith("Frame Time:"):
                frame_time = float(stripped.split(":")[1].strip())
                past_header = True
                continue
            if past_header and stripped:
                lines.append(stripped)

    return lines, frame_time


def main() -> None:
    parser = argparse.ArgumentParser(description="Send BVH data over UDP")
    parser.add_argument("--bvh", required=True, help="Path to BVH file")
    parser.add_argument("--host", default="127.0.0.1", help="Target host")
    parser.add_argument("--port", type=int, default=1118, help="Target UDP port")
    parser.add_argument("--fps", type=float, default=0, help="Override send rate (0 = use BVH frame time)")
    parser.add_argument("--loop", action="store_true", help="Loop the BVH data indefinitely")
    parser.add_argument("--downsample", type=int, default=1, help="Send every Nth frame")
    args = parser.parse_args()

    motion_lines, frame_time = _read_motion_lines(args.bvh)
    if not motion_lines:
        print(f"No motion data found in {args.bvh}")
        return

    # Apply downsampling
    if args.downsample > 1:
        motion_lines = motion_lines[:: args.downsample]
        frame_time *= args.downsample

    # Override fps if specified
    if args.fps > 0:
        frame_time = 1.0 / args.fps

    n_floats = len(motion_lines[0].split())
    print(f"Loaded {len(motion_lines)} frames, {n_floats} floats/frame, "
          f"sending at {1.0/frame_time:.1f} fps to {args.host}:{args.port}")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    target = (args.host, args.port)

    try:
        iteration = 0
        while True:
            start = time.monotonic()
            for i, line in enumerate(motion_lines):
                t0 = time.monotonic()
                sock.sendto(line.encode("utf-8"), target)
                elapsed = time.monotonic() - t0
                sleep = frame_time - elapsed
                if sleep > 0:
                    time.sleep(sleep)

            iteration += 1
            elapsed_total = time.monotonic() - start
            print(f"  Iteration {iteration}: {len(motion_lines)} frames in {elapsed_total:.2f}s")

            if not args.loop:
                break
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        sock.close()


if __name__ == "__main__":
    main()
