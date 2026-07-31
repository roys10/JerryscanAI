# PatchCore G01: U2-Net black background

Place the matching local artifacts here:

- `G01.ckpt` (ignored by Git)
- `G01.metadata.json` (versioned after review)

The required preprocessing contract is [`rembg_u2net_black_v1.json`](../../training/preprocessing/configs/rembg_u2net_black_v1.json). Live preprocessing resolves the versioned U2-Net parent contract and requires the ignored `models/preprocessing/rembg/u2net.onnx` artifact. Calibration is pending and must not be fabricated.
