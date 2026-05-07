import sys
from config import load_settings, mock_metadata
from vectordb_operations.milvus_dump import LocalDumpManager

#######################################
# Configure

# Load settings
settings = load_settings()

# Overwrite settings if necessary
settings.milvus_host = "localhost"
settings.milvus_port = 19530

# Set dump parameters
collection_name = "test_vm2"  # Name of the collection in Milvus
root_dump_dir = "./dumps_collection"  # Root directory where dumps are stored
dump_name = "test_vm_dump"  # Optional: custom name for the dump (default: collection_name_dateTag)


#######################################
# Start Export Process

# Load helper
dump_mgr = LocalDumpManager(
    settings,
    mock_metadata(collection_name),
)

# Export collection dump
dump_mgr.export_collection_dump(
    root_dump_dir=root_dump_dir,
    dump_name=dump_name,  # if None: collection_name_dateTag
)
# Export the collection dump with extra info in the name
dump_mgr.export_metadata(
    root_dump_dir=root_dump_dir,
    dump_name=dump_name,  # if None: collection_name_dateTag
)
