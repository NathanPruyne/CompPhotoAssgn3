import skimage
import numpy as np

import argparse

def load_image(path):
    imarr = skimage.io.imread(path)

    #Normalize to (0, 1)
    if imarr.dtype == np.uint8:
        imarr = imarr.astype(np.float32) / 255
    elif imarr.dtype == np.uint16:
        imarr = imarr.astype(np.float32) / 65535
    else:
        imarr = imarr.astype(np.float32)

    #Drop alpha channel if present
    if imarr.ndim == 3 and imarr.shape[2] == 4:
        imarr = imarr[:, :, :3]

    return imarr

def difference(im1, im2):
    if im1.shape != im2.shape:
        raise ValueError(f"Image shapes do not match: {im1.shape} vs {im2.shape}")

    return im1 - im2

def print_stats(diff):
    abs_diff = np.abs(diff)
    mse = np.mean(diff ** 2)
    psnr = np.inf if mse == 0 else 10 * np.log10(1 / mse)

    print(f"Mean abs diff: {np.mean(abs_diff):.6f}")
    print(f"Max abs diff:  {np.max(abs_diff):.6f}")
    print(f"MSE:           {mse:.6f}")
    print(f"PSNR:          {psnr:.2f} dB")

    if diff.ndim == 3:
        for channel in range(diff.shape[2]):
            print(f"  Channel {channel} mean abs diff: {np.mean(abs_diff[:, :, channel]):.6f}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("image1", type=str, help="First image")
    parser.add_argument("image2", type=str, help="Second image")
    parser.add_argument("output", type=str, help="Output difference image")
    parser.add_argument("--scale", type=float, default=1.0, help="Amplification factor for the difference")
    parser.add_argument("--signed", action="store_true", help="Show signed difference (gray = no change, bright = image1 brighter, dark = image2 brighter)")

    args = parser.parse_args()

    im1 = load_image(args.image1)
    im2 = load_image(args.image2)

    diff = difference(im1, im2)
    print_stats(diff)

    if args.signed:
        vis = 0.5 + 0.5 * args.scale * diff
    else:
        vis = args.scale * np.abs(diff)

    vis = np.clip(vis, 0, 1)

    skimage.io.imsave(args.output, (vis * 255).astype(np.uint8))
