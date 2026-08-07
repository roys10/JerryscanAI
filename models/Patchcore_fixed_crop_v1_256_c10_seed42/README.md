# Fixed-crop PatchCore

This run's reviewed metadata is already included. Add its matching `G01.ckpt`,
which stays local and is ignored by Git. The tracked `model.json` applies the same fixed G01 crop
used to generate the training dataset, then letterboxes that crop to 1024 x
1024 before PatchCore inference.

Use original production-camera frames. A resized upload can be too small for
the recorded crop and will correctly return `WRONG_INPUT`. Its provisional raw
PatchCore image-score threshold is 36, the ceiling above the maximum score
observed across 994 normal `split_v2` validation images. This produced zero
observed validation false positives; defect recall is not yet validated.
