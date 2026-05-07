import pytest
from pydantic import ValidationError
from config import (
    MilvusCollectionMetadata,
    MilvusSchema,
    MilvusIndex,
    FieldParams,
    IndexParams,
)


def test_valid_metadata_full():
    """Test that a valid metadata dictionary passes validation."""
    valid_data = {
        "collection_name": "test_collection",
        "schema": {
            "collection_name": "test_collection",
            "num_shards": 1,
            "aliases": ["test_alias"],
            "description": "A test collection",
            "fields": [
                {
                    "name": "text_id",
                    "type": 21,
                    "params": {"max_length": 36},
                    "is_primary": True,
                },
                {"name": "embedding", "type": 101, "params": {"dim": 384}},
            ],
            "enable_dynamic_field": True,
        },
        "index": {
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 128},
        },
    }
    metadata = MilvusCollectionMetadata(**valid_data)
    assert metadata.collection_name == "test_collection"
    assert len(metadata.collection_schema.fields) == 2


def test_multiple_fields_mixed_types():
    """Test a schema with many fields of different types (INT64, VARCHAR, FLOAT_VECTOR)."""
    data = {
        "collection_name": "complex_coll",
        "schema": {
            "collection_name": "complex_coll",
            "num_shards": 2,
            "aliases": ["a1", "a2"],
            "fields": [
                {"name": "pk", "type": 5, "params": {}, "is_primary": True},  # INT64
                {"name": "label", "type": 21, "params": {"max_length": 128}},  # VARCHAR
                {"name": "vector_small", "type": 101, "params": {"dim": 128}},
                {"name": "vector_large", "type": 101, "params": {"dim": 1536}},
                {"name": "metadata_json", "type": 23, "params": {}},  # JSON
            ],
        },
        "index": {"index_type": "HNSW", "metric_type": "COSINE", "params": {}},
    }
    metadata = MilvusCollectionMetadata(**data)
    assert len(metadata.collection_schema.fields) == 5
    assert metadata.collection_schema.fields[1].params.max_length == 128
    assert metadata.collection_schema.fields[3].params.dim == 1536


def test_optional_defaults():
    """Test that optional fields receive their default values correctly."""
    minimal_data = {
        "collection_name": "min_coll",
        "schema": {
            "collection_name": "min_coll",
            "num_shards": 1,
            "aliases": [],
            "fields": [{"name": "id", "type": 5, "params": {}, "is_primary": True}],
        },
        "index": {"index_type": "FLAT", "metric_type": "IP"},
    }
    metadata = MilvusCollectionMetadata(**minimal_data)
    assert metadata.collection_schema.description == ""
    assert metadata.collection_schema.auto_id is False
    assert metadata.collection_schema.enable_dynamic_field is False
    assert metadata.collection_schema.mmap_enabled is False


def test_collection_name_mismatch():
    """Test that the validator raises an error if collection names don't match."""
    mismatched_data = {
        "collection_name": "outer",
        "schema": {
            "collection_name": "inner",
            "num_shards": 1,
            "aliases": [],
            "fields": [],
        },
        "index": {"index_type": "FLAT", "metric_type": "IP"},
    }
    with pytest.raises(ValueError, match="does not match schema collection name"):
        MilvusCollectionMetadata(**mismatched_data)


def test_invalid_num_shards_negative():
    """Test that num_shards must be positive."""
    with pytest.raises(ValidationError):
        MilvusSchema(collection_name="t", aliases=[], num_shards=-1, fields=[])


def test_index_params_missing():
    """Test index creation with missing params block."""
    index_data = {"index_type": "FLAT", "metric_type": "L2"}
    index = MilvusIndex(**index_data)
    assert index.params.nlist is None


def test_index_params_invalid_nlist():
    """Test that nlist must be a positive integer."""
    with pytest.raises(ValidationError):
        IndexParams(nlist=0)
    with pytest.raises(ValidationError):
        IndexParams(nlist=-10)


def test_extra_fields_ignored():
    """Test that extra keys in the metadata JSON are ignored (default Pydantic behavior)."""
    data = {
        "collection_name": "test",
        "unknown_top_level_key": "ignore me",
        "schema": {
            "collection_name": "test",
            "num_shards": 1,
            "aliases": [],
            "fields": [],
            "unexpected_schema_key": 123,
        },
        "index": {"index_type": "FLAT", "metric_type": "IP"},
    }
    metadata = MilvusCollectionMetadata(**data)
    assert metadata.collection_name == "test"
    # Verify we can't access the unknown keys on the model
    assert not hasattr(metadata, "unknown_top_level_key")


def test_field_params_mixed_nulls():
    """Test FieldParams with partial nulls."""
    params = FieldParams(max_length=None, dim=512)
    assert params.max_length is None
    assert params.dim == 512
