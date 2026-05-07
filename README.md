# VectorDB Operations

A robust toolkit for managing Milvus vector database operations, featuring high-performance bulk data utilities, automated collection management, and a local development sandbox.

## Table of Contents
- [Key Features](#key-features)
- [Installation](#installation)
- [Configuration](#configuration)
- [Local Sandbox (Docker Compose)](#local-sandbox-docker-compose)
- [Library Usage (Python API)](#library-usage-python-api)
- [CLI Utility: milvus_dump](#cli-utility-milvus_dump)
- [Stability & Performance](#stability--performance)
- [Development & Testing](#development--testing)

## Key Features

- **Pythonic API**: Clean interface for common Milvus tasks (storing, querying, and managing embeddings).
- **High-Performance CLI**: `milvus_dump` utility for fast export/import using Parquet files.
- **Memory Efficient**: Built-in support for `--mmap` to optimize memory usage for large indices.
- **Validation**: Pydantic-powered configuration and metadata validation.
- **Developer Sandbox**: Pre-configured Docker Compose environment (Milvus + Attu + MinIO) for rapid testing.
- **Rich Observability**: CLI output with real-time progress bars and styled logging.

## Installation

### Using uv (Recommended)

```bash
uv sync
source .venv/bin/activate
```

### Using pip

```bash
python3.12 -m venv .venv # Windows: py -3.12 -m venv .venv
source .venv/bin/activate # Windows:  .venv\Scripts\activate
pip install -e .
```

## Configuration

The package uses Pydantic for configuration. It automatically loads baseline values from `config/env.example`
and overrides them with any values found in a `.env` file in the project root.

| Variable | Default | Description |
|----------|---------|-------------|
| `MILVUS_HOST` | `localhost` | Milvus server address |
| `MILVUS_PORT` | `19530` | Milvus server port |
| `MINIO_ENDPOINT` | `http://localhost:9000` | Object storage for bulk writer |
| `MINIO_ACCESS_KEY` | `minioadmin` | MinIO access key |
| `MINIO_SECRET_KEY` | `minioadmin` | MinIO secret key |

**Note:** The `milvus_dump` CLI utility dynamically adopts these values as its default arguments.
You can see the current active milvus connection (host:port) by running `milvus_dump {export,import} --help`.

## Local Sandbox (Docker Compose)

A `docker-compose.yml` is included to spin up a fully self-contained Milvus environment for local development and testing. It starts:

| Service | Description | Port |
|---------|-------------|------|
| **etcd** | Metadata store for Milvus | — |
| **MinIO** | Object storage (S3 API & Console) | 9000 (API), 9001 (console) |
| **Milvus** | Vector database (standalone) | 19530 |
| **Attu** | Web UI for browsing collections | 8000 |

### Start the sandbox

```bash
docker compose up -d
```

Wait until Milvus is healthy (usually ~30 s):

```bash
docker compose ps          # all services should show "healthy" or "running"
```

Open the Attu web UI at **http://localhost:8000** — connect to `milvus:19530` (no credentials needed for the sandbox).

### Stop and clean up

```bash
docker compose down        # stop containers, keep volumes
docker compose down -v     # stop containers and delete all data
```

## Library Usage (Python API)

The `vector_db` module provides a high-level wrapper for common operations.

```python
import numpy as np
from vectordb_operations.vector_db import (
    create_collection,
    store_embedding,
    store_embedding_bulk,
    query_embedding,
    query_similar_by_vector,
    get_all_ids_in_namespace,
)

# --- create a collection with 384-dimensional embeddings ---
collection = create_collection("my_collection", dim=384)

# --- store a single embedding ---
vec = np.random.rand(384).tolist()
store_embedding("doc-001", vec, collection)

# --- store multiple embeddings at once ---
ids   = [f"doc-{i:03d}" for i in range(2, 1001)]
vecs  = [np.random.rand(384).tolist() for _ in ids]
store_embedding_bulk(ids, vecs, collection)

# --- retrieve embeddings by ID ---
result = query_embedding(["doc-001", "doc-002"], "my_collection")
print(result["text_ids"])       # ['doc-001', 'doc-002']
print(len(result["embeddings"])) # 2

# --- find top-5 nearest neighbours for a query vector ---
query_vec = np.random.rand(384).tolist()
hits = query_similar_by_vector(query_vec, "my_collection", top_k=5)
for hit in hits:
    print(hit["text_id"], hit["distance"])

# --- list all IDs stored in the collection ---
all_ids = get_all_ids_in_namespace("my_collection")
print(f"{len(all_ids)} embeddings in collection")
```

## CLI Utility: `milvus_dump`

The `milvus_dump` utility handles bulk data mobility.

### Export a Collection

```bash
milvus_dump export -c my_collection -o ./my_dumps --mmap
```

### Import a Collection

```bash
milvus_dump import -d ./my_dumps/my_collection_dump -m ./my_dumps/metadata.json # (force replace --drop-existing )
```

The import uses the S3 API port from MinIO. If this port is has not been exposed from the container
one can find its address by running:

```bash
# Get IP for the milvus database
docker inspect -f '{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}' milvus-standalone
# Get IP for MinIO
docker inspect -f '{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}' milvus-minio
```

where `milvus-...` are the containers name.
The output will look like `172.##.#.##` which can then be added to the `.env` file.


### Enable Autocomplete

```bash
eval "$(register-python-argcomplete milvus_dump)"
```

## Stability & Performance

- **Memory Efficiency (`--mmap`)**: Using the `--mmap` flag allows Milvus to map index files from disk instead of loading them into the heap. This is highly recommended for large collections to prevent memory spikes.
- **Rich Logging**: Uses the `rich` library for real-time progress tracking and clear status markers.

## Development & Testing

Run the test suite using `pytest`

```bash
pytest
```

# FAQ

## "error during connect"
If you're on Windows, remember to start up the Docker Desktop software manually.
