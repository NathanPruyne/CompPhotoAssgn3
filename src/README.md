
## Files

| File | Purpose |
|---|---|
| `src/bilateral.py` | Basic bilateral, joint bilateral, detail transfer, and shadow/specularity masking (Problem 1) |
| `src/gradient_domain.py` | Gradient/divergence/Laplacian operators, conjugate gradient Poisson solver, and flash/no-flash gradient fusion (Problem 2) |
| `src/diff.py` | Difference images and error statistics between two images |

## Problem 1: Bilateral filtering (`src/bilateral.py`)

```
python src/bilateral.py INPUT OUTPUT [--flash FLASH] [options]
```

`INPUT` is the ambient (no-flash) image. The four algorithms are selected by flags:

| Result | Flags |
|---|---|
| Basic bilateral (A<sup>Base</sup>) | *(none)* |
| Joint bilateral (A<sup>NR</sup>) | `--flash FLASH` |
| Detail transfer (A<sup>Detail</sup>) | `--flash FLASH --detail` |
| Masked merge (A<sup>Final</sup>) | `--flash FLASH --detail --mask` |

Options:

| Option | Default | Meaning |
|---|---|---|
| `--spatial_dev` | 16 | σ<sub>s</sub>, spatial Gaussian standard deviation (pixels) |
| `--intensity_dev` | 0.15 | σ<sub>r</sub>, intensity Gaussian standard deviation (images normalized to [0, 1]) |
| `--minmax_lambda` | 0.01 | λ padding on the min/max intensity when creating segments |
| `--detail_eps` | 0.02 | ε in the detail transfer ratio (F + ε) / (F<sup>Base</sup> + ε) |
| `--tau_shadow` | 0.05 | Shadow threshold on linearized F − A luminance |
| `--tau_spec` | 0.95 | Specularity threshold on linearized flash luminance |
| `--iso_diff` | 8 | Ambient ISO / flash ISO (lamp pair: 1600 / 200) |
| `--resize` | 1.0 | Resize both inputs by this factor first (e.g. `0.25` for large camera images) |

The same σ<sub>s</sub> and σ<sub>r</sub> are used for every bilateral filter within one run, including F<sup>Base</sup> for detail transfer and A<sup>Base</sup> for masking.

Example (lamp pair):

```bash
python src/bilateral.py data/lamp/lamp_ambient.tif out/lamp_basic.png --spatial_dev 16 --intensity_dev 0.15
python src/bilateral.py data/lamp/lamp_ambient.tif out/lamp_joint.png --flash data/lamp/lamp_flash.tif
python src/bilateral.py data/lamp/lamp_ambient.tif out/lamp_detail.png --flash data/lamp/lamp_flash.tif --detail
python src/bilateral.py data/lamp/lamp_ambient.tif out/lamp_final.png --flash data/lamp/lamp_flash.tif --detail --mask --tau_shadow 0.005
```

Runtime grows with image size and with 1/σ<sub>r</sub> (the number of intensity segments), so small σ<sub>r</sub> on full-size images can take several minutes.

## Problem 2: Gradient-domain processing (`src/gradient_domain.py`)

```
python src/gradient_domain.py INPUT OUTPUT [--flash FLASH] [--fuse] [options]
```

Without `--fuse`, the script differentiates `INPUT` and re-integrates it with the Poisson solver, which should reproduce the original image. With `--fuse`, `INPUT` is the ambient image and `--flash` is the flash image.

Options:

| Option | Default | Meaning |
|---|---|---|
| `--eps` | 1e-3 | CGD stops when the residual norm falls below this |
| `--iters` | 1000 | Maximum CGD iterations (per color channel) |
| `--boundary` | `ambient` | Dirichlet boundary values: `ambient`, `flash`, `average`, or `zero` |
| `--init` | `zero` | CGD initialization: `ambient`, `flash`, `average`, or `zero` |
| `--sigma` | 40 | σ in the saturation weight w<sub>s</sub> = tanh(σ(Φ′ − τ<sub>s</sub>)) |
| `--tau_s` | 0.9 | τ<sub>s</sub>, flash saturation threshold |
| `--save_grads` | off | Also save ∇a, ∇Φ′, and ∇Φ* (fusion only) |
| `--grad_scale` | 5 | Amplification for saved gradient images |
| `--resize` | 1.0 | Resize both inputs by this factor first (e.g. `0.25` for large camera images) |

`flash` and `average` for `--boundary`/`--init` require `--fuse`. For the re-integration test, keep `--boundary ambient` so the boundary comes from the original image.

With `--save_grads`, an output `out/fused.png` also writes `out/fused_{amb,flash,fused}_grad_{x,y}.png`. In these images gray is zero gradient, brighter is positive, and darker is negative, scaled by `--grad_scale`.

Examples (museum pair):

```bash
# Differentiate and re-integrate
python src/gradient_domain.py data/museum/museum_ambient.png out/museum_reint.png --iters 3000

# Fusion, saving gradient fields
python src/gradient_domain.py data/museum/museum_ambient.png out/museum_fused.png \
    --flash data/museum/museum_flash.png --fuse --boundary average --init average --iters 2000 --save_grads
```

## Problem 3: Own flash/no-flash pairs

The same scripts are used on my own photos in `data/myrtle` and `data/van`.

## Difference images (`src/diff.py`)

```
python src/diff.py IMAGE1 IMAGE2 OUTPUT [--scale S] [--signed]
```

Saves |IMAGE1 − IMAGE2| (or, with `--signed`, a signed difference where gray is no change, brighter means IMAGE1 is brighter, and darker means IMAGE2 is brighter), amplified by `--scale`. Prints mean/max absolute difference, MSE, PSNR, and per-channel mean absolute difference.

```bash
python src/diff.py out/lamp_joint.png out/lamp_basic.png out/joint_minus_basic.png --signed --scale 4
```
