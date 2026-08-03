# Manufacturing frontend

This React/Vite application is the operator interface for the JerryscanAI
manufacturing backend. It is separate from the research Model Lab frontend.

The current runtime supports original G01 camera images and exactly one active
PatchCore model folder. The backend applies that model's preprocessing; do not
upload a preprocessed derivative. Until a real-fault validation threshold is
approved, successful inspections appear as `SHADOW / UNDECIDED`. Invalid
inputs or unavailable artifacts require `REVIEW` rather than producing PASS.

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

## Checks

```powershell
npm run lint
npm run build
```
