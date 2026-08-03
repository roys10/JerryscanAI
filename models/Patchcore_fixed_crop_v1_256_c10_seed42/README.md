# Fixed-crop PatchCore

This run's reviewed metadata is already included. Add its matching `G01.ckpt`,
which stays local and is ignored by Git. The tracked `model.json` applies the same fixed G01 crop
used to generate the training dataset, then letterboxes that crop to 1024 x
1024 before PatchCore inference.

Use original production-camera frames. A resized upload can be too small for
the recorded crop and will correctly return `REVIEW`. The threshold remains
`null`, so this model cannot emit an automatic PASS/FAIL decision.
