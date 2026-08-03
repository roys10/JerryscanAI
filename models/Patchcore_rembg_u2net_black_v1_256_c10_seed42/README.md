# U2Net black-background PatchCore

This run's metadata is already included; add its matching `G01.ckpt`. Live inference runs the same
U2Net mask and alignment used by the gray variant, but composites the aligned
jerrycan on black before PatchCore.

The backend prefers a local `u2net.onnx` and otherwise uses
`models/preprocessing/rembg/u2net.onnx`. Copy the ONNX file beside `model.json`
when the folder must work by itself. Learned/local artifacts are ignored by
Git. The threshold is intentionally `null`, so results are
`SHADOW / UNDECIDED`.
