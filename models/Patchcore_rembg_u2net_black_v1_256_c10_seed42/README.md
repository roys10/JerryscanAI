# U2Net black-background PatchCore

This model set contains one PatchCore checkpoint and metadata pair for each
production camera, G01 through G04. Live inference runs one shared U2Net mask
and alignment pipeline for every original image, composites the aligned
jerrycan on black, and sends it to the checkpoint for that camera angle.

The backend prefers a local `u2net.onnx` and otherwise uses
`models/preprocessing/rembg/u2net.onnx`. Copy the ONNX file beside `model.json`
when the folder must work by itself. Learned/local artifacts are ignored by
Git. `model.json` records the SHA-256, byte size, metadata, and decision
threshold independently for G01-G04. G01's provisional raw PatchCore threshold
of 34 came from its normal validation set. G02-G04 currently use an explicitly
marked temporary value of 34; calibrate those angles on their own validation
scores before treating their PASS/FAIL decisions as qualified. Defect recall
is not yet validated for any angle.
