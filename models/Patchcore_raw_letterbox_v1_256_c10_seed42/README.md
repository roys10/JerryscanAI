# Raw-letterbox PatchCore

This folder is one complete local inference choice. Its reviewed training
metadata is already included; add the matching `G01.ckpt`, which stays local
and is ignored by Git. The tracked `model.json` tells the backend to
letterbox each original G01 camera image onto a 1024 x 1024 gray canvas before
running PatchCore.

Select it by setting `JERRYSCAN_MODEL_FOLDER` to this directory. The threshold
is intentionally `null`, so results stay `SHADOW / UNDECIDED`.
