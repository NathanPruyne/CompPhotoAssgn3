import skimage
import numpy as np
import cv2
from scipy.interpolate import interpn

import argparse

def luminance(imarr):
    weights = np.array([0.2126, 0.7152, 0.0722]).reshape(3, 1, 1)
    return np.sum(imarr * weights, axis=0)

def linearize(imarr):
    return np.where(imarr <= 0.04045, imarr / 12.92, ((imarr + 0.055) / 1.055) ** 2.4)

def bilateral(ambient, flash=None, spatial_dev=16, intensity_dev=0.15, minmax_lambda=0.01, detail_transfer=False, detail_eps=0.02, mask=False, tau_shadow=0.05, tau_spec=0.95, iso_diff=8):

    if flash is None:
        flash = ambient

    minI = np.min(flash) - minmax_lambda
    maxI = np.max(flash) + minmax_lambda
    
    nb_segs = int(np.ceil((maxI - minI)/intensity_dev))
    i_j_vals = np.linspace(minI, maxI, nb_segs + 1)
    
    C, H, W = ambient.shape
    rows, cols = np.indices((H, W))


    A_NR = np.zeros_like(ambient, dtype=float)

    for channel in range(C):
        I_A = ambient[channel]
        I_F = flash[channel]
        accum = np.zeros((nb_segs + 1, H, W))
        for j, i_j in enumerate(i_j_vals):
            G_j = np.exp((-1 *  (I_F - i_j)**2) / (2 * intensity_dev ** 2))
            K_j = cv2.GaussianBlur(G_j, (0, 0), sigmaX=spatial_dev, borderType=cv2.BORDER_CONSTANT)
            H_j = G_j * I_A
            H_starj = cv2.GaussianBlur(H_j, (0, 0), sigmaX=spatial_dev, borderType=cv2.BORDER_CONSTANT)
            J_j = H_starj / np.maximum(K_j, 1e-10)
            accum[j] = J_j
        x_i = np.stack([I_F, rows, cols], axis=-1)
        A_NR[channel] = interpn((i_j_vals, np.arange(H), np.arange(W)), accum, x_i)

    if detail_transfer:

        F_base = bilateral(flash, spatial_dev=spatial_dev, intensity_dev=intensity_dev, minmax_lambda=minmax_lambda, detail_transfer=False)

        A_detail = A_NR * (flash + detail_eps) / (F_base + detail_eps)

        if mask:
            ambient_lum = luminance(linearize(ambient)) * 1 / iso_diff
            flash_lum = luminance(linearize(flash))
            shadow_pix = np.where((flash_lum - ambient_lum) <= tau_shadow, 1, 0)
            spec_pix = np.where(flash_lum > tau_spec, 1, 0)
            M = np.logical_or(shadow_pix, spec_pix)[None].astype(float)
            A_base = bilateral(ambient, spatial_dev=spatial_dev, intensity_dev=intensity_dev, minmax_lambda=minmax_lambda, detail_transfer=False)
            final = (1 - M) * A_detail + M * A_base
        else:
            final = A_detail

    else:

        final = A_NR


    return final


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("input", type=str, help="Input (no-flash) image")
    parser.add_argument("output", type=str, help="Output file")
    parser.add_argument("--flash", type=str, help="Flash (guide) image")
    parser.add_argument("--spatial_dev", type=int, default=16, help="Standard deviation of spatial Gaussian")
    parser.add_argument("--intensity_dev", type=float, default=0.15, help="Standard deviation of intensity Gaussian")
    parser.add_argument("--detail", action="store_true", help="Add flash-to-ambient detail transfer")
    parser.add_argument("--minmax_lambda", type=float, default=0.01, help="Padding added to min/max intensity when creating segments")
    parser.add_argument("--detail_eps", type=float, default=0.02, help="Epsilon used in detail transfer")
    parser.add_argument("--mask", action="store_true", help="Apply shadow and specularity masking (requires --detail)")
    parser.add_argument("--tau_shadow", type=float, default=0.05, help="Shadow threshold on linear flash - ambient luminance")
    parser.add_argument("--tau_spec", type=float, default=0.95, help="Specularity threshold on linear flash luminance")
    parser.add_argument("--iso_diff", type=float, default=8, help="Ratio of ambient ISO to flash ISO")
    parser.add_argument("--resize", type=float, default=1.0, help="Resize factor applied to input images before processing (e.g. 0.25)")

    args = parser.parse_args()

    imarr = skimage.io.imread(args.input)
    if args.resize != 1.0:
        imarr = cv2.resize(imarr, None, fx=args.resize, fy=args.resize, interpolation=cv2.INTER_AREA)
    if imarr.shape[2] == 3: #Change to CxWxH
        imarr = np.transpose(imarr, axes=(2, 0, 1))

    #Normalize to (0, 1)
    orig_dtype = imarr.dtype
    if orig_dtype == np.uint8:
        imarr = np.astype(imarr, np.float32) / 255
    
    print("Applying bilateral filter")

    if args.flash:
        flasharr = skimage.io.imread(args.flash)
        if args.resize != 1.0:
            flasharr = cv2.resize(flasharr, None, fx=args.resize, fy=args.resize, interpolation=cv2.INTER_AREA)
        if flasharr.shape[2] == 3: #Change to CxWxH
            flasharr = np.transpose(flasharr, axes=(2, 0, 1))
    
        #Normalize to (0, 1)
        flash_dtype = flasharr.dtype
        if flash_dtype == np.uint8:
            flasharr = np.astype(flasharr, np.float32) / 255
    else:
        flasharr = None


    bilat = bilateral(imarr, flash=flasharr, spatial_dev=args.spatial_dev, intensity_dev=args.intensity_dev,
                      minmax_lambda=args.minmax_lambda, detail_transfer=args.detail, detail_eps=args.detail_eps,
                      mask=args.mask, tau_shadow=args.tau_shadow, tau_spec=args.tau_spec, iso_diff=args.iso_diff)

    bilat = np.transpose(bilat, (1, 2, 0))
    bilat = np.clip(bilat, 0, 1)

    if orig_dtype == np.uint8:
        bilat = np.astype(bilat * 255, np.uint8)

    print("Saving")

    skimage.io.imsave(args.output, bilat)
