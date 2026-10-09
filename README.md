# MetaboAgent

MetaboAgent is an agentic metabolomics workflow that combines literature-grounded
candidate discovery, data preprocessing, feature selection, multi-objective panel
optimization, scientific visualization, and evidence-to-decision reporting.

## Repository layout

```text
backend/     FastAPI service and streaming job interface
frontend/    React + TypeScript web application
src/         Phase 0-4 agents, analysis tools, memory, and reporting code
data/        Versioned workflow/SOP definitions only
demo_data/   Small reviewer-facing example dataset
storage/     Local knowledge assets (not committed)
output/      Generated analysis artifacts (not committed)
config.yaml  Runtime and workflow configuration
```

The repository intentionally does not include user datasets, generated results,
logs, caches, model files, local vector databases, credentials, or large reference
knowledge assets. The one file under `demo_data/` is the documented reviewer
example and is included so the demo selector works after a fresh clone.

## Requirements

- Python 3.10 or 3.11
- Node.js 18+
- R and the packages used by `src/tools/visualization/r/` for R-backed figures
- An OpenAI-compatible API key

AutoGluon and several scientific packages are sizeable. A fresh installation can
take several minutes and may require platform-specific build tools.

## Backend setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

Add provider credentials to `.env`, then install or build the local knowledge
assets described in [DATA_ASSETS.md](DATA_ASSETS.md).

Start the API from the repository root:

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

The health endpoint is `http://localhost:8000/health`; interactive API
documentation is available at `http://localhost:8000/docs`.

## Frontend setup

```bash
cd frontend
npm ci
cp .env.example .env
npm run dev
```

By default, the frontend is served at `http://localhost:5173` and communicates
with the backend on port `8000` through the Vite development proxy. For a
reverse-proxy subpath deployment, set `VITE_BASE_PATH=/metaboagent/` in
`frontend/.env` and build the frontend; leave `VITE_API_BASE_URL` and
`VITE_WS_BASE_URL` empty when the proxy serves both paths from the same origin.

The upload page includes the versioned `demo_data/liver_cancer_demo.csv`
example. It uses `id` as the sample ID, `group` as the binary outcome, and `1`
as the positive class.

## Configuration

`config.yaml` controls workflow paths, model behavior, optimization objectives,
visualization policy, and memory settings. Runtime outputs are written under
`output/`, `storage/`, and `backend/storage/`; these paths are ignored by Git.

Never commit `.env` files, API keys, patient-level datasets, or generated job
artifacts.

## Frontend quality checks

```bash
cd frontend
npm run check
npm run build
```

## License

No open-source license has been assigned yet. Add an appropriate `LICENSE` file
before publishing if reuse or redistribution should be permitted.
