# Raw-letterbox PatchCore

This folder is one complete local inference choice. Its reviewed training
metadata is already included; add the matching `G01.ckpt`, which stays local
and is ignored by Git. The tracked `model.json` tells the backend to
letterbox each original G01 camera image onto a 1024 x 1024 gray canvas before
running PatchCore.

Select it by setting `JERRYSCAN_MODEL_FOLDER` to this directory. Its
provisional raw PatchCore image-score threshold is 35, the ceiling above the
maximum score observed across 994 normal `split_v2` validation images. This
produced zero observed validation false positives; defect recall is not yet
validated.
