import pytest
import os
from config import load_example_metadata, load_settings
from vectordb_operations.milvus_dump import LocalDumpManager


@pytest.fixture(scope="module")
def schema_config():
    """Loads the environment variables via Pydantic before tests run."""
    return load_example_metadata()


@pytest.fixture(scope="module")
def settings():
    """Loads the environment variables via Pydantic before tests run."""
    return load_settings()


@pytest.fixture
def milvus_utils(schema_config, settings, monkeypatch):
    # Setup
    mgr = LocalDumpManager(settings, schema_config)
    root_dump_dir = "./dumps_collections"
    dump_name = "test_dump"

    yield mgr, root_dump_dir, dump_name

    # Interactively confirm export (simulate user input "yes" for confirmation)
    monkeypatch.setattr("builtins.input", lambda _: "yes")
    # Cleanup ( also on fail )
    mgr.clean_bucket()
    mgr.clean_dump(os.path.join(root_dump_dir, dump_name))
    mgr.clean_dump(os.path.join(root_dump_dir, dump_name + "_export"))
    try:
        # Remove folder if empty
        os.rmdir(root_dump_dir)
    except OSError:
        pass


def test_full_dump_and_import_workflow(milvus_utils):
    dump_mgr, root_dump_dir, dump_name = milvus_utils

    # Create Dummy index
    dump_mgr.drop_collection()
    dump_mgr.create_collection()

    # Create Dummy data
    dump_dir = dump_mgr.create_dummy_dump(
        root_dump_dir=root_dump_dir,
        dump_name=dump_name,
        size=10_000,
        segment_size_mb=128,
    )

    # Upload to MinIO and import to Milvus
    dump_mgr.import_collection_dump(dump_data_dir=dump_dir)
    dump_mgr.setup_index()

    # Check export
    export_path = dump_mgr.export_collection_dump(
        root_dump_dir=root_dump_dir, dump_name=dump_name + "_export"
    )
    assert export_path is not None
