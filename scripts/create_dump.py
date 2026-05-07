from config import load_example_metadata, load_settings
from vectordb_operations.milvus_dump import LocalDumpManager
from vectordb_operations.milvus_manager import MilvusCollMgrError

metadata = load_example_metadata()
settings = load_settings()
# schema_config = load_dummy_schema()  # Use dummy schema for testing

# Overwrite settings if necessary
settings.milvus_host = "localhost"
settings.milvus_port = 19530

# Define collection name in Milvus
metadata.collection_name = "gnd_de_snowflakearctic_dummy_bulk"
metadata.collection_schema.collection_name = "gnd_de_snowflakearctic_dummy_bulk"

# Load helper
dump_mgr = LocalDumpManager(settings, metadata)

# Set Workspace directory where all dumps will be created (default: ./milvus_dump)
root_dump_dir = "./dumps_collection"
# Optional: specify a custom name for the dump (default: collection_name_dateTag)
dump_name = "my_dump"

# Export collection dump
# default output_dir is : ROOT_DUMP_DIR/collection_name_dateTag
try:
    # Try to setup index first, if it fails
    dump_mgr.setup_index(use_existing=True)
except MilvusCollMgrError as e:
    # Create index on the collection before exporting
    dump_mgr.setup_index()
dump_mgr.export_collection_dump(root_dump_dir=root_dump_dir, dump_name=dump_name)
