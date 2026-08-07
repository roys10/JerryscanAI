# U2Net black-background PatchCore

This run's metadata is already included; add its matching `G01.ckpt`. Live inference runs the same
U2Net mask and alignment used by the gray variant, but composites the aligned
jerrycan on black before PatchCore.

The backend prefers a local `u2net.onnx` and otherwise uses
`models/preprocessing/rembg/u2net.onnx`. Copy the ONNX file beside `model.json`
when the folder must work by itself. Learned/local artifacts are ignored by
Git. Its provisional raw PatchCore image-score threshold is 34, the ceiling
above the maximum score observed across 994 normal `split_v2` validation
images. This produced zero observed validation false positives; defect recall
is not yet validated.
