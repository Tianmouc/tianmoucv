"""Low-overhead Tianmouc camera preview."""

import argparse
import threading
import time
from queue import Queue

import cv2
import numpy as np

from tianmoucv.camera.sdk_utils import TMstreamingDataReader, put_latest


def torch_camera_ops():
    """Import the optional reconstruction stack when it is selected."""
    import torch
    from tianmoucv.proc.reconstruct import sd_gradients, poisson_blending, sd_gray_preview

    return torch, sd_gradients, poisson_blending, sd_gray_preview


def to_tensor(array, torch):
    return torch.from_dlpack(np.ascontiguousarray(array))


def to_numpy(tensor):
    return np.from_dlpack(tensor.detach().cpu().contiguous())


def raw_rod_preview(rawdata, scale=8):
    """Visualize TD, SDL and SDR as signed-color images."""
    images = []
    for channel in rawdata:
        signed = channel.astype(np.int16, copy=False)
        image = np.zeros((*channel.shape, 3), dtype=np.uint8)
        image[..., 0] = np.clip(-signed * scale, 0, 255)
        image[..., 2] = np.clip(signed * scale, 0, 255)
        images.append(image)
    return np.concatenate(images, axis=1)


def process_aop(input_queue, output_queue, iterations):
    torch, sd_gradients, poisson_blending, preview = torch_camera_ops()
    while True:
        sdl, sdr = input_queue.get()
        with torch.inference_mode():
            gx, gy = sd_gradients(to_tensor(np.stack((sdl, sdr)), torch))
            gray = preview(poisson_blending(gx, gy))
        put_latest(output_queue, gray)


def process_cop(input_queue, output_queue, iterations):
    torch, sd_gradients, poisson_blending, _ = torch_camera_ops()
    while True:
        rgb, rod = input_queue.get()
        _, sdl, sdr = rod
        with torch.inference_mode():
            gx, gy = sd_gradients(to_tensor(np.stack((sdl, sdr)), torch).float() / 128)
            hdr = poisson_blending(gx, gy, srcimg=to_tensor(rgb, torch).div(255.0),
                                   mask_rgb=True, mask_th=24)

        hdr = np.clip(to_numpy(hdr) * 255.0, 0, 255).astype(np.uint8)
        put_latest(output_queue, cv2.cvtColor(hdr, cv2.COLOR_RGB2BGR))


def start_worker(target, *args):
    threading.Thread(target=target, args=args, daemon=True).start()


def show_result(name, queue):
    if queue.empty():
        return False
    cv2.imshow(name, queue.get_nowait())
    return True


def parse_args():
    parser = argparse.ArgumentParser(description="Preview a Tianmouc camera")
    parser.add_argument("--id", type=int, default=0, help="zero-based camera id")
    parser.add_argument("--lossless", action="store_true")
    parser.add_argument(
        "--aop-mode", choices=("fast", "raw", "poisson", "off"), default="fast"
    )
    parser.add_argument(
        "--cop-mode", choices=("rgb", "hdr", "off"), default="rgb"
    )
    parser.add_argument("--aop-fps", type=float, default=20.0)
    parser.add_argument("--cop-fps", type=float, default=30.0)
    parser.add_argument("--iterations", type=int, default=5, help="compatibility option; Poisson now uses a direct solver")
    parser.add_argument("--threads", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    cv2.setNumThreads(args.threads)

    aop_input = Queue(maxsize=1)
    cop_input = Queue(maxsize=1)
    aop_output = Queue(maxsize=1)
    cop_output = Queue(maxsize=1)

    if args.aop_mode == "poisson":
        import torch
        torch.set_num_threads(args.threads)
        start_worker(process_aop, aop_input, aop_output, args.iterations)

    if args.cop_mode == "hdr":
        import torch
        torch.set_num_threads(args.threads)
        start_worker(process_cop, cop_input, cop_output, args.iterations)

    reader = TMstreamingDataReader(device_id=args.id, latest=not args.lossless)
    reader.setExposure(aoptime=1200, coptime=15000, aopG=1, copG=1)

    latest_rod = None
    next_aop = 0.0
    next_cop = 0.0
    next_ui_poll = 0.0

    try:
        with reader:
            while True:
                frame = reader(copy=False)
                if frame is None:
                    continue

                is_rod, _, data = frame
                now = time.monotonic()
                updated = False

                if is_rod:
                    latest_rod = data
                    if args.aop_mode != "off" and now >= next_aop:
                        next_aop = now + 1.0 / args.aop_fps
                        if args.aop_mode == "fast":
                            cv2.imshow("AOP", reader.render_rod_preview())
                            updated = True
                        elif args.aop_mode == "raw":
                            cv2.imshow("AOP", raw_rod_preview(data))
                            updated = True
                        else:
                            put_latest(
                                aop_input, (data[1].copy(), data[2].copy())
                            )

                elif args.cop_mode != "off" and now >= next_cop:
                    next_cop = now + 1.0 / args.cop_fps
                    if args.cop_mode == "rgb":
                        cv2.imshow("COP", reader.render_cone_preview())
                        updated = True
                    elif latest_rod is not None:
                        rod = tuple(channel.copy() for channel in latest_rod)
                        put_latest(cop_input, (data.copy(), rod))

                updated |= show_result("AOP", aop_output)
                updated |= show_result("COP", cop_output)

                if updated or now >= next_ui_poll:
                    next_ui_poll = now + 1.0 / 60.0
                    if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                        break
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
