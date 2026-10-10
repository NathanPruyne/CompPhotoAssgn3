import numpy as np
import skimage
import cv2
from scipy.signal import convolve2d
from tqdm import tqdm

import argparse
import os

def save_gradient(prefix, I_x, I_y, scale=5):
    #I_x, I_y are CxHxW; zero gradient maps to gray, positive brighter, negative darker
    for name, grad in (("x", I_x), ("y", I_y)):
        vis = np.clip(0.5 + scale * np.transpose(grad, (1, 2, 0)), 0, 1)
        skimage.io.imsave(f"{prefix}_{name}.png", np.round(vis * 255).astype(np.uint8))

def gradient(imarr):
    I_x = np.diff(imarr, axis=1, append=imarr[:, -1:])
    I_y = np.diff(imarr, axis=0, append=imarr[-1:, :])
    return I_x, I_y

def divergence(I_x, I_y):
    I_xx = np.diff(I_x, axis=1, prepend=0)
    I_yy = np.diff(I_y, axis=0, prepend=0)
    return I_xx + I_yy

def laplacian(imarr):
    kernel = [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
    return convolve2d(imarr, kernel, mode="same", boundary="fill", fillvalue=0)

def CGD(D, I_star_init, I_star_boundary, eps=1e-3, iters=1000):
    B = np.zeros_like(D)
    B[1:-1, 1:-1] = 1
    I_star = B * I_star_init + (1 - B) * I_star_boundary
    r = B * (D - laplacian(I_star))
    d = r
    delta_new = np.vdot(r, r)
    n = 0
    pbar = tqdm(total=iters, desc="CGD")
    while np.sqrt(np.vdot(r, r)) > eps and n < iters:
        q = laplacian(d)
        eta = delta_new / np.vdot(d, q)
        I_star = I_star + B * (eta * d)
        r = B * (r - eta * q)
        delta_old = delta_new
        delta_new = np.vdot(r, r)
        beta = delta_new / delta_old
        d = r + beta * d
        n += 1
        pbar.update(1)
        pbar.set_postfix(residual=f"{np.sqrt(delta_new):.2e}")
    pbar.close()
    return I_star

def get_inital(ambient, type, flash=None):
    if type == "ambient":
        return ambient
    elif type == "flash":
        if flash is None:
            raise ValueError("Asking for flash boundary with no flash image")
        return flash
    elif type == "average":
        if flash is None:
            raise ValueError("Asking for average boundary with no flash image")
        return (flash + ambient) / 2
    elif type == "zero":
        return np.zeros_like(ambient)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("input", type=str, help="Input (no-flash) image")
    parser.add_argument("output", type=str, help="Output file")
    parser.add_argument("--flash", type=str, help="Flash (guide) image")
    parser.add_argument("--fuse", action="store_true", help="Perform fusion or just derivative/integral")
    parser.add_argument("--eps", type=float, default=1e-3, help="CGD convergence threshold")
    parser.add_argument("--iters", type=int, default=1000, help="Maximum number of CGD iterations")
    parser.add_argument("--sigma", type=float, default=40, help="Steepness of saturation weight tanh")
    parser.add_argument("--tau_s", type=float, default=0.9, help="Flash saturation threshold")
    parser.add_argument("--boundary", type=str, default="ambient", choices=["ambient", "flash", "average", "zero"], help="Image used for Dirichlet boundary values")
    parser.add_argument("--init", type=str, default="zero", choices=["ambient", "flash", "average", "zero"], help="Image used to initialize CGD")
    parser.add_argument("--save_grads", action="store_true", help="Save ambient, flash, and fused gradient fields as images next to the output (fusion only)")
    parser.add_argument("--grad_scale", type=float, default=5, help="Amplification factor for saved gradient images")
    parser.add_argument("--resize", type=float, default=1.0, help="Resize factor applied to input images before processing (e.g. 0.25)")

    args = parser.parse_args()

    imarr = skimage.io.imread(args.input)
    imarr = imarr[:, :, :3] #Drop alpha channel if present
    if args.resize != 1.0:
        imarr = cv2.resize(imarr, None, fx=args.resize, fy=args.resize, interpolation=cv2.INTER_AREA)
    imarr = np.transpose(imarr, axes=(2, 0, 1)) #Change to CxHxW

    #Normalize to (0, 1)
    orig_dtype = imarr.dtype
    if orig_dtype == np.uint8:
        imarr = np.astype(imarr, np.float32) / 255

    if args.fuse: #do fusion stuffs
        if args.flash is None:
            raise ValueError("Can't fuse without flash image")
    
        flasharr = skimage.io.imread(args.flash)
        flasharr = flasharr[:, :, :3] #Drop alpha channel if present
        if args.resize != 1.0:
            flasharr = cv2.resize(flasharr, None, fx=args.resize, fy=args.resize, interpolation=cv2.INTER_AREA)
        flasharr = np.transpose(flasharr, axes=(2, 0, 1)) #Change to CxHxW
    
        #Normalize to (0, 1)
        orig_dtype = flasharr.dtype
        if orig_dtype == np.uint8:
            flasharr = np.astype(flasharr, np.float32) / 255
        
        result = np.zeros_like(imarr)
        grads = {name: np.zeros_like(imarr) for name in ("amb_x", "amb_y", "flash_x", "flash_y", "fused_x", "fused_y")}

        for channel in range(imarr.shape[0]):

            amb_x, amb_y = gradient(imarr[channel])
            flash_x, flash_y = gradient(flasharr[channel])

            M = np.abs(flash_x * amb_x + flash_y * amb_y) / (np.sqrt(flash_x ** 2 + flash_y ** 2) * np.sqrt(amb_x ** 2 + amb_y ** 2) + 1e-8)

            w_s = np.tanh(args.sigma * (flasharr[channel] - args.tau_s))
            w_s = (w_s - np.min(w_s)) / (np.max(w_s) - np.min(w_s)) #Normalize

            phi_star_x = w_s * amb_x + (1 - w_s) * (M * flash_x + (1 - M) * amb_x)
            phi_star_y = w_s * amb_y + (1 - w_s) * (M * flash_y + (1 - M) * amb_y)

            grads["amb_x"][channel], grads["amb_y"][channel] = amb_x, amb_y
            grads["flash_x"][channel], grads["flash_y"][channel] = flash_x, flash_y
            grads["fused_x"][channel], grads["fused_y"][channel] = phi_star_x, phi_star_y

            D = divergence(phi_star_x, phi_star_y)
            I_star_init = get_inital(imarr[channel], args.init, flash=flasharr[channel])
            I_star_boundary = get_inital(imarr[channel], args.boundary, flash=flasharr[channel])
            result[channel] = CGD(D, I_star_init, I_star_boundary, eps=args.eps, iters=args.iters)

        if args.save_grads:
            prefix = os.path.splitext(args.output)[0]
            for name in ("amb", "flash", "fused"):
                save_gradient(f"{prefix}_{name}_grad", grads[f"{name}_x"], grads[f"{name}_y"], scale=args.grad_scale)



    else: #derivative and reintegration
        result = np.zeros_like(imarr)
        for channel in range(imarr.shape[0]):
            D = divergence(*gradient(imarr[channel]))
            I_star_init = get_inital(imarr[channel], args.init)
            I_star_boundary = get_inital(imarr[channel], args.boundary)
            result[channel] = CGD(D, I_star_init, I_star_boundary, eps=args.eps, iters=args.iters)

    result = np.transpose(result, (1, 2, 0))
    result = np.clip(result, 0, 1)

    if orig_dtype == np.uint8:
        result = np.astype(result * 255, np.uint8)

    print("Saving")

    skimage.io.imsave(args.output, result)

    