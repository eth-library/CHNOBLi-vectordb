import argparse
import sys
from vectordb_operations.milvus_dump import LocalDumpManager
from vectordb_operations.milvus_manager import MilvusCollectionManager
from config import load_settings, load_example_metadata, mock_metadata


def run_info_bucket(args, settings):
    metadata = mock_metadata("info")
    dump_mgr = LocalDumpManager(settings, metadata)
    print(f"Fetching info for bucket: {settings.minio_bucket}")
    minio = dump_mgr._get_minio_handler()
    minio.stats()


def run_info_collections(args, settings):
    metadata = mock_metadata("info")
    mgr = MilvusCollectionManager(settings, metadata)
    mgr.stats()


def run_info(args):
    settings = load_settings()
    subcommand = getattr(args, "info_subcommand", None)

    if subcommand == "bucket":
        run_info_bucket(args, settings)
    elif subcommand == "collections":
        run_info_collections(args, settings)
    else:
        print("--- Milvus Collections ---")
        run_info_collections(args, settings)
        print("\n--- MinIO Bucket ---")
        run_info_bucket(args, settings)


def run_clean(args):
    settings = load_settings()
    if args.collection:
        metadata = mock_metadata(args.collection)
    else:
        metadata = load_example_metadata()
    
    dump_mgr = LocalDumpManager(settings, metadata)
    dump_mgr.clean_bucket()


def main():
    parser = argparse.ArgumentParser(
        description="MinIO Bucket Cleanup and Information Utility",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Info Command
    info_parser = subparsers.add_parser("info", help="Display information")
    info_subparsers = info_parser.add_subparsers(dest="info_subcommand", help="Sub-information to display")
    
    info_subparsers.add_parser("bucket", help="Show MinIO bucket statistics")
    info_subparsers.add_parser("collections", help="List Milvus collections")
    
    info_parser.set_defaults(func=run_info)

    # Clean Command
    clean_parser = subparsers.add_parser("clean", help="Clean MinIO bucket prefixes")
    clean_parser.add_argument(
        "-c",
        "--collection",
        help="Name of the collection whose MinIO files should be cleaned (defaults to example metadata name)",
    )
    clean_parser.set_defaults(func=run_clean)

    # Default to info if no command is provided
    parser.set_defaults(func=run_info)

    args = parser.parse_args()
    
    try:
        args.func(args)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
