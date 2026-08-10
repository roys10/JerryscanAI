# Manufacturing frontend

This React/Vite application is the operator interface for the JerryscanAI
manufacturing backend. It is separate from the research Model Lab frontend.

The runtime supports exactly one active PatchCore model-set folder and derives
its camera buttons from that folder's declared angles. Upload one original
image for every required camera; the backend applies the shared preprocessing
and matching angle checkpoint. Do not upload preprocessed derivatives.
Successful inference returns `PASS` below that angle's raw-score threshold and
`FAIL` at or above it. Threshold calibration provenance is recorded per angle
in `model.json`; defect recall is not yet validated. Invalid images show
`Wrong input`; unavailable model artifacts are reported as an API error rather
than as a defective jerrycan.

Successful results provide three separate views: **Defect Location** draws the
main-style red anomaly contour, **Anomaly Map** shows the fixed-scale heatmap,
and **Preprocessing Mask** shows the segmentation mask used to isolate and
align the jerrycan. The PASS/FAIL badge is deliberately simple. Result details
show a **Quality score** from 0% to 100%, where 100% means no measured anomaly
and the failure threshold is 70%. The score is calculated as
`clamp(100 - 30 * raw_score / raw_threshold, 0, 100)`, so the backend's raw
decision threshold maps exactly to 70%. This is a relative quality index, not
confidence, probability, or accuracy. Model Configuration shows the model name
and the operator threshold as `70%`.

## Start locally

Start the backend from the repository root first; see the
[main project README](../README.md#manufacturing-application). Then run:

```powershell
cd frontend
npm install
npm run dev
```

Open the Vite URL printed in the terminal (normally
`http://127.0.0.1:5173`). The frontend calls `http://localhost:8000` by
default. To use another backend address, set `VITE_BACKEND_URL` before starting
or building the frontend.

For the remote VM deployment on HTTP port 80, use the VM address without an
explicit port, for example `VITE_BACKEND_URL=http://192.0.2.10`. Vite embeds
this value when the frontend is built. A frontend served over HTTPS cannot call
an HTTP backend because browsers block mixed content; that setup requires TLS
on the backend or an HTTPS reverse proxy.

## Checks

```powershell
npm run lint
npm run build
```
