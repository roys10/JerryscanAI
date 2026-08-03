# U2Net gray-background PatchCore

This run's metadata is already included; add its matching `G01.ckpt`. Live inference runs U2Net,
keeps and aligns the largest jerrycan mask, replaces the background with gray,
then runs PatchCore.

The backend first looks for `u2net.onnx` in this folder and otherwise uses the
shared `models/preprocessing/rembg/u2net.onnx`. To copy a fully portable folder,
place `u2net.onnx` beside `model.json`. The checkpoint and ONNX weight are
ignored by Git. The missing validation threshold keeps results in
`SHADOW / UNDECIDED`.
