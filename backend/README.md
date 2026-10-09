# MetaboAgent API

The backend is a FastAPI application that exposes health, upload, demo-dataset,
job, result, artifact-download, and streaming endpoints for the MetaboAgent
workflow.

Start it from the repository root:

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

API documentation is available at `http://localhost:8000/docs` after startup.

The service loads the repository-root `.env` explicitly. Copy `.env.example` to
`.env` and provide credentials locally; never commit that file.
