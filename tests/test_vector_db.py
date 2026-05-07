import pytest
import numpy as np
from unittest.mock import patch
from vectordb_operations.vector_db import(store_embedding,
                                          ids_in_namespace,
                                          query_embedding,
                                          compare_vector_to_text_ids,
                                          compare_vector_to_text_ids_multiplexed)

class FakeCollection:
    def __init__(self, name, rows=None):
        self.name = name
        self.rows = rows or {}  # text_id -> embedding
        self.loaded = False

    def insert(self, data):
        ids, embeddings = data
        for tid, emb in zip(ids, embeddings):
            self.rows[tid] = emb

    def flush(self):
        pass

    def load(self):
        self.loaded = True

    def release(self):
        self.loaded = False

    def query(self, expr, output_fields=None):
        # very naive parser: text_id in ["a","b"]
        ids = expr.split("[", 1)[1].split("]", 1)[0]
        ids = [x.strip().strip('"') for x in ids.split(",") if x.strip()]
        return [
            {
                "text_id": tid,
                "embedding": self.rows[tid],
            }
            for tid in ids
            if tid in self.rows
        ]

    def search(self, data, anns_field, param, limit, output_fields):
        query_vec = np.array(data[0])
        hits = []
        for tid, emb in self.rows.items():
            dist = np.linalg.norm(query_vec - np.array(emb))
            hits.append(
                type(
                    "Hit",
                    (),
                    {
                        "distance": float(dist),
                        "entity": {"text_id": tid},
                    },
                )
            )
        hits = sorted(hits, key=lambda h: h.distance)[:limit]
        return [hits]

    def query_iterator(self, expr, output_fields, batch_size):
        rows = [{"text_id": tid} for tid in self.rows]
        return FakeIterator(rows, batch_size)


class FakeIterator:
    def __init__(self, rows, batch_size):
        self.rows = rows
        self.batch_size = batch_size
        self.i = 0

    def next(self):
        if self.i >= len(self.rows):
            return []
        batch = self.rows[self.i : self.i + self.batch_size]
        self.i += self.batch_size
        return batch

    def close(self):
        pass
    
    
@pytest.fixture
def fake_collection():
    return FakeCollection(
        "embeddings",
        rows={
            "a": [1.0, 0.0],
            "b": [0.0, 1.0],
        },
    )


@pytest.fixture
def milvus_patches(fake_collection):
    with patch("vectordb_operations.vector_db.connections.connect"), \
         patch("vectordb_operations.vector_db.utility.has_collection", return_value=True), \
         patch("vectordb_operations.vector_db.Collection", return_value=fake_collection):
        yield fake_collection

def test_store_embedding_inserts_data(milvus_patches):
    store_embedding("c", [0.5, 0.5], milvus_patches)
    assert "c" in milvus_patches.rows
    
def test_ids_in_namespace_partial_match(milvus_patches):
    result = ids_in_namespace("embeddings", ["a", "x"])
    assert result == {"a": True, "x": False}
    
def test_query_embedding_returns_embeddings(milvus_patches):
    res = query_embedding(["a", "b"], "embeddings")
    assert res["text_ids"] == ["a", "b"]
    assert len(res["embeddings"]) == 2
    
def test_compare_vector_cosine(milvus_patches):
    vec = [1.0, 0.0]
    res = compare_vector_to_text_ids(
        vec,
        "embeddings",
        ["a", "b"],
        distance_metric="cosine",
    )

    by_id = {r["text_id"]: r["distance"] for r in res}
    assert by_id["a"] < by_id["b"]
    
def test_multiplexed_empty_candidates(milvus_patches):
    input_data = [
        {"query_id": 1, "reference_text_ids": []},
    ]
    vectors = [[1.0, 0.0]]

    res = compare_vector_to_text_ids_multiplexed(
        input_data, vectors, "embeddings"
    )

    assert res == [{"query_id": 1, "distances": []}]
