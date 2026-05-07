from config import load_settings, load_metadata
from vectordb_operations.milvus_dump import LocalDumpManager

#######################################
# Configure

# Import settings
settings = load_settings()

# Overwrite settings if necessary
settings.milvus_host = "localhost"
settings.milvus_port = 19530

# Dump info
dump_dir = "./dumps_collection/test_vm_20260417/"
metadata_path = f"{dump_dir}/test_vm2_metadata.json"

#######################################
# Start Upload Process

# Import metadata
metadata = load_metadata(metadata_path)
# Load helper
dump_mgr = LocalDumpManager(settings, metadata)

# Uncomment if the collection already exists
# WARNING: This will delete all data in the collection, use with caution!
# dump_mgr.drop_collection()  # Drop existing collection if it exists

# Import collection dump
dump_mgr.create_collection()  # Create new collection based on the schema
dump_mgr.import_collection_dump(
    "./dumps_collection/gnd_de_snowflakearctic_dummy_bulk_20260416/"
)
# Create index on the collection before exporting
dump_mgr.setup_index()
