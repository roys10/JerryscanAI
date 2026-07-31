# PatchCore G01: U2-Net gray background

Place the matching local artifacts here:

- `G01.ckpt` (ignored by Git)
- `G01.metadata.json` (versioned after review)

The required preprocessing contract is [`rembg_u2net_gray_v1.json`](../../training/preprocessing/configs/rembg_u2net_gray_v1.json). Live preprocessing also requires the ignored `models/preprocessing/rembg/u2net.onnx` artifact with the hash recorded in the preprocessing model registry. Calibration is pending and must not be fabricated.
