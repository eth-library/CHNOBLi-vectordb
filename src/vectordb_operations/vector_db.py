"""
vector_db.py

This module provides functions to interact with a Milvus vector database for storing and querying text embeddings. It connects to a Milvus instance, ensures the required collection exists, and exposes functions for inserting and retrieving embeddings by text ID.

Environment Variables:
- MILVUS_HOST: Hostname for Milvus (default: 'milvus')
- MILVUS_PORT: Port for Milvus (default: '19530')

Collection Schema:
- Collection name: 'embeddings'
- Fields:
    - text_id (VARCHAR, primary key): Unique identifier for the text (max_length=36)
    - embedding (FLOAT_VECTOR): Embedding vector (dim=384)

Functions:
- store_embedding(text_id, embedding_vector): Store an embedding in the collection.
- query_embedding(text_id): Retrieve the embedding for a given text ID.
"""

from copy import deepcopy
from numbers import Number
from typing import Any
import h5py

import numpy as np
from pymilvus import (  # type: ignore
    Collection,
    CollectionSchema,
    Connections,
    DataType,
    FieldSchema,
    connections,
    utility,
)
from scipy.spatial.distance import cosine, euclidean
from typing_extensions import List

MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"

def set_milvus_cfg(host: str, port: str):
    global MILVUS_HOST
    global MILVUS_PORT
    MILVUS_HOST = host
    MILVUS_PORT = port

def connect_milvus() -> Connections:
    # Connect to Milvus
    connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
    return connections


def create_collection(
    collection_name: str,
    dim: int = 768,
    index_type: str = "IVF_SQ8",
    metric_type: str = "COSINE",
    nlist: int = 128,
) -> Collection:
    connect_milvus()

    if not utility.has_collection(collection_name):
        fields = [
            FieldSchema(
                name="text_id",
                dtype=DataType.VARCHAR,
                max_length=36,
                is_primary=True,
                auto_id=False,
            ),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
        ]
        schema = CollectionSchema(fields, "Embedding collection")
        collection = Collection(name=collection_name, schema=schema)
        collection.create_index(
            field_name="embedding",
            index_params={
                "index_type": index_type,
                "metric_type": metric_type,
                "params": {"nlist": nlist},
            },
        )
    else:
        collection = Collection(collection_name)
    return collection


# Function to store embeddings
def store_embedding(
    text_id: str, embedding_vector: List[float], collection: Collection
) -> None:
    """
    Store an embedding vector in the Milvus collection.

    Args:
        text_id (str): Unique identifier for the text.
        embedding_vector: The embedding vector to store (list or numpy array).
        collection: The Milvus collection object.
    """
    collection.insert([[text_id], [embedding_vector]])
    collection.flush()


def store_embedding_bulk(
    text_ids: List[str], embedding_vectors: List[List[float]], collection: Collection
) -> None:
    """
    Store a list of vectors in the Milvus collection.

    Args:
        text_ids (list[str]): Unique identifiers for the texts.
        embedding_vectors (list[list[float]]): The embedding vectors to store.
        collection: The Milvus collection object.
    """
    collection.insert([text_ids, embedding_vectors])
    collection.flush()


# Function to query embeddings
def query_embedding(
    text_ids: list[str], collection_name: str
) -> dict | None:
    """
    Retrieve the embedding vector(s) for one or more text IDs from the Milvus collection.

    Args:
        text_ids (str | list[str]): Single text ID or list of IDs.
        collection_name (str): Name of the Milvus collection.

    Returns:
        list or None: A list of results (each containing "embedding" and "text_id"),
                      or None if nothing was found.
    """

    valid_ids_map = ids_in_namespace(collection_name, text_ids)
    valid_ids = [tid for tid, exists in valid_ids_map.items() if exists]
    quoted_ids = ", ".join(f'"{tid}"' for tid in valid_ids)
    expr = f"text_id in [{quoted_ids}]"

    connect_milvus()
    collection = Collection(collection_name)
    collection.load()

    result: list[Any] = collection.query(
        expr=expr, output_fields=["text_id", "embedding"]
    )

    result = deepcopy(result)
    collection.release()

    if not result:
        return None

    ids = []
    embeddings = []

    for r in result:
        ids.append(r["text_id"])
        embeddings.append(r["embedding"])

    return {"text_ids": ids, "embeddings": embeddings}


def query_similar_by_vector(
    vector: list[float],
    collection_name: str,
    top_k: int = 5,
    metric_type: str = "COSINE",
    nprobe: int = 10,
) -> list[dict[str, str | float]]:
    """
    Query Milvus for the top_k closest embeddings to a given input vector.

    Args:
        vector (list[float]): Input embedding vector.
        collection_name (str): Milvus collection to query.
        top_k (int): Number of most similar results to return.
        metric_type (str): Distance metric used at search time; must match the
            index metric. Defaults to "COSINE" to align with create_collection.
        nprobe (int): Number of units to query during search (IVF param).

    Returns:
        list of dict: Each dict contains text_id and distance.
    """
    connect_milvus()
    collection = Collection(collection_name)
    collection.load()

    search_params = {"metric_type": metric_type, "params": {"nprobe": nprobe}}
    search_results = collection.search(
        data=[vector],
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        output_fields=["text_id"],
    )

    collection.release()

    hits = search_results[0]
    return [
        {"text_id": hit.entity.get("text_id"), "distance": hit.distance} for hit in hits
    ]


def get_all_ids_in_namespace(
    collection_name: str, batch_size: int = 10000
) -> list[str]:
    """
    Retrieve all text_ids from the given Milvus collection in batches.

    Args:
        collection_name (str): The Milvus collection name.
        batch_size (int): Number of records to fetch per batch.

    Returns:
        list[str]: All text_ids in the collection.
    """
    connect_milvus()
    collection = Collection(collection_name)
    collection.load()

    all_ids: list[str] = []

    # use iterator to stream results
    iterator = collection.query_iterator(
        expr="",  # no filter
        output_fields=["text_id"],
        batch_size=batch_size,
    )

    while True:
        batch = iterator.next()
        if not batch:
            break
        all_ids.extend(row["text_id"] for row in batch)

    iterator.close()
    collection.release()
    return all_ids


def ids_in_namespace(collection_name: str, text_ids: list[str]) -> dict[str, bool]:
    """
    Check which of the given text_ids exist in the Milvus collection.

    Args:
        collection_name (str): The Milvus collection name.
        text_ids (list[str]): List of text_ids to check.

    Returns:
        dict[str, bool]: Mapping of text_id to a boolean indicating existence in the namespace.
    """
    connect_milvus()
    collection = Collection(collection_name)
    collection.load()

    quoted_ids = ", ".join(f'"{tid}"' for tid in text_ids)
    expr = f"text_id in [{quoted_ids}]"
    results = collection.query(expr=expr, output_fields=["text_id"])
    existing_ids = {res["text_id"] for res in results}

    collection.release()

    return {tid: tid in existing_ids for tid in text_ids}


def compare_vector_to_text_ids(
    vector: list[float],
    collection_name: str,
    target_text_ids: list[str],
    distance_metric: str = "cosine",
) -> list[dict[str, str | Number]]:
    """
    Compare a given vector against embeddings for a specific list of existing text_ids.

    Args:
        vector (list[float]): The query embedding vector.
        collection_name (str): The Milvus collection name.
        target_text_ids (list[str]): List of text_ids to compare against.
        distance_metric: str: distance metric to be used "eucledian" or "cosine"

    Returns:
        list[dict]: Each dict contains 'text_id' and its 'distance' from the query vector.
    """
    valid_ids_map = ids_in_namespace(collection_name, target_text_ids)
    valid_ids = [tid for tid, exists in valid_ids_map.items() if exists]

    if not valid_ids:
        return []  # No valid IDs to compare

    connect_milvus()
    collection = Collection(collection_name)
    collection.load()

    quoted_ids = ", ".join(f'"{tid}"' for tid in valid_ids)
    expr = f"text_id in [{quoted_ids}]"

    results = collection.query(expr=expr, output_fields=["text_id", "embedding"])

    collection.release()

    distances = []
    for item in results:
        emb = item["embedding"]
        tid = item["text_id"]
        if distance_metric == "cosine":
            distance = cosine(vector, emb)
        else:
            distance = euclidean(vector, emb)  # type: ignore
        distances.append({"text_id": tid, "distance": distance})

    return distances


def compare_vector_to_text_ids_multiplexed(  # type: ignore
    input: dict, 
    vectors: list[list[float]], 
    collection_name:str, 
    distance_metric:str = "cosine"
) -> list[dict]:
    """
    Compare query embeddings to candidate embeddings stored in Milvus,
    restricted by reference_text_ids per query.

    input: list of dicts with keys "query_id": int, "query_text": str, "reference_text_ids": list[str]
    vectors: list of embeddings (aligned with input order)
    collection_name: Milvus collection
    ditance_metric: "cosine" | "l2" | "ip"
    """
    connect_milvus()
    collection = Collection(collection_name)
    collection.load()

    # Collect all candidate IDs across all queries
    all_ids = set()
    for item in input:
        all_ids.update(item["reference_text_ids"])
    all_ids = list(all_ids)  # type: ignore

    if not all_ids:
        return [{"query_id": item["query_id"], "distances": []} for item in input]

    # single Milvus query
    expr = f"text_id in {all_ids}"
    candidate_rows = collection.query(expr=expr, output_fields=["text_id", "embedding"])

    id_to_emb = {
        row["text_id"]: np.array(row["embedding"], dtype=np.float32)
        for row in candidate_rows
    }

    results = []

    # compute distances
    for i, item in enumerate(input):
        query_id = item["query_id"]
        candidate_ids = item["reference_text_ids"]
        query_vec = np.array(vectors[i], dtype=np.float32)

        # Keep only candidates that exist in DB
        cand_embs = []
        cand_ids = []
        for tid in candidate_ids:
            if tid in id_to_emb:
                cand_ids.append(tid)
                cand_embs.append(id_to_emb[tid])
        if not cand_embs:
            results.append({"query_id": query_id, "distances": []})
            continue

        cand_embs = np.vstack(cand_embs)  # type: ignore

        # Distance computation
        if distance_metric == "cosine":
            q = query_vec / np.linalg.norm(query_vec)
            C = cand_embs / np.linalg.norm(cand_embs, axis=1, keepdims=True)
            sims = np.dot(C, q)
            dists = 1 - sims
        elif distance_metric == "l2":
            dists = np.linalg.norm(cand_embs - query_vec, axis=1)
        elif distance_metric == "ip":
            dists = -np.dot(cand_embs, query_vec)  # higher dot = closer
        else:
            raise ValueError(f"Unsupported distance metric: {distance_metric}")

        # Rank results
        ranked = sorted(zip(cand_ids, dists), key=lambda x: x[1])
        results.append(
            {
                "query_id": query_id,
                "distances": [
                    {"reference_text_id": tid, "distance": float(dist)}
                    for tid, dist in ranked
                ],
            }
        )

    collection.release()
    return results

def export_collection(
    collection_name: str,
    batch_size: int = 1000,
    output_path: str = "output/export.h5",
) -> None:
    """
    Export a Milvus collection to an HDF5 file by streaming batches from Milvus
    and appending them incrementally to disk. Memory usage is O(batch_size), not
    O(total entities), so arbitrarily large collections are supported.
 
    The output file contains two resizable datasets:
      - "text_id":   variable-length UTF-8 strings, shape (N,)
      - "embedding": float32 vectors,               shape (N, dim)
 
    Args:
        collection_name (str): Name of the Milvus collection to export.
        batch_size (int): Number of entities fetched from Milvus per iteration.
        output_path (str): Destination .h5 file path.
    """
    connect_milvus()
    col = Collection(collection_name)
    col.load()
 
    total = col.num_entities
    print(f"Total entities to export: {total}")
 
    # Use query_iterator so Milvus pages internally — we never hold more than
    # one batch in Python memory at a time.
    iterator = col.query_iterator(
        expr="",
        output_fields=["text_id", "embedding"],
        batch_size=batch_size,
    )
 
    # String dtype for HDF5 variable-length UTF-8
    str_dtype = h5py.special_dtype(vlen=str)
 
    exported = 0
    with h5py.File(output_path, "w") as hf:
        id_ds = None
        emb_ds = None
 
        while True:
            batch = iterator.next()
            if not batch:
                break
 
            batch_ids = [row["text_id"] for row in batch]
            batch_embs = np.array(
                [row["embedding"] for row in batch], dtype=np.float32
            )
            n, dim = batch_embs.shape
 
            if id_ds is None:
                # Create resizable datasets on the first batch so we can infer dim.
                id_ds = hf.create_dataset(
                    "text_id",
                    shape=(0,),
                    maxshape=(None,),
                    dtype=str_dtype,
                    chunks=(min(batch_size, 10_000),),
                )
                emb_ds = hf.create_dataset(
                    "embedding",
                    shape=(0, dim),
                    maxshape=(None, dim),
                    dtype=np.float32,
                    chunks=(min(batch_size, 10_000), dim),
                )
 
            # Append by resizing then writing the new slice.
            current = id_ds.shape[0]
            id_ds.resize(current + n, axis=0)
            emb_ds.resize(current + n, axis=0)
            id_ds[current : current + n] = batch_ids
            emb_ds[current : current + n] = batch_embs
 
            exported += n
            print(f"  {exported}/{total} exported...")
 
    iterator.close()
    col.release()
    connections.disconnect("default")
    print(f"\nDone → {output_path}")
 
 
def import_collection(
    collection_name: str,
    input_path: str = "output/export.h5",
    batch_size: int = 1000,
) -> None:
    """
    Import a collection from an HDF5 export file into Milvus by streaming
    slices from disk and inserting them in batches. Memory usage is
    O(batch_size), not O(total entities), so arbitrarily large files are
    supported.

    Args:
        collection_name (str): Name of the Milvus collection to import into.
        input_path (str): Path to the .h5 export file produced by export_collection().
        batch_size (int): Number of entities to read from disk and insert per batch.
    """
    with h5py.File(input_path, "r") as hf:
        id_ds = hf["text_id"]
        emb_ds = hf["embedding"]

        total: int = id_ds.shape[0]
        dim: int = emb_ds.shape[1]

    print(f"Detected embedding dim: {dim}")
    print(f"Entities to import: {total}")

    collection = create_collection(collection_name, dim=dim)
    collection.load()

    # Re-open for streaming so we never load the full file into RAM.
    with h5py.File(input_path, "r") as hf:
        id_ds = hf["text_id"]
        emb_ds = hf["embedding"]

        for offset in range(0, total, batch_size):
            end = min(offset + batch_size, total)

            # HDF5 slice reads only the requested rows from disk.
            # h5py returns variable-length strings as bytes in Python 3 — decode them.
            batch_ids: list[str] = [
                v.decode("utf-8") if isinstance(v, bytes) else v
                for v in id_ds[offset:end].tolist()
            ]
            batch_embs: list[list[float]] = emb_ds[offset:end].tolist()

            store_embedding_bulk(batch_ids, batch_embs, collection)
            print(f"  {end}/{total} inserted...")

    print(f"Import complete → '{collection_name}' ({total} entities)")


