import argparse
import argcomplete
import logging
import os
import sys
from pathlib import Path

# Only import settings and metadata functions when needed to speed up CLI startup time
from config import load_settings

# Get logger for this module
logger = logging.getLogger(__name__)


class MilvusDumpCliError(Exception):
    """Custom exception for CLI errors that should not show a traceback."""

    def __init__(self, message):
        super().__init__(message)
        # stacklevel=2 makes the logger report the line where the exception was RAISED
        logger.error(message, stacklevel=2)


class CustomHelpFormatter(argparse.HelpFormatter):
    def __init__(self, prog):
        super().__init__(prog, max_help_position=42)


def add_common_arguments(parser, settings):
    """Helper to add shared connection arguments to a parser group."""
    group = parser.add_argument_group("Connection & Performance Settings")
    group.add_argument(
        "--host",
        default=settings.milvus_host,
        help=f"Milvus host (default: {settings.milvus_host})",
    )
    group.add_argument(
        "--port",
        type=int,
        default=settings.milvus_port,
        help=f"Milvus port (default: {settings.milvus_port})",
    )
    group.add_argument(
        "--minio-host",
        default=settings.minio_host,
        help=f"MinIO host (default: {settings.minio_host})",
    )
    group.add_argument(
        "--minio-port",
        type=int,
        default=settings.minio_port,
        help=f"MinIO port (default: {settings.minio_port})",
    )
    group.add_argument(
        "--mmap",
        action="store_true",
        help="Enable memory-mapped files (mmap) to reduce memory spikes during load",
    )


def run_export(args):
    """Handles the export command."""
    # Defer heavy imports
    from config import mock_metadata, load_settings
    from vectordb_operations.milvus_dump import LocalDumpManager

    print(f"Starting export for collection: {args.collection}")

    settings = load_settings()
    settings.milvus_host = args.host
    settings.milvus_port = args.port
    settings.minio_host = args.minio_host
    settings.minio_port = args.minio_port

    metadata = mock_metadata(args.collection)

    # Enable mmap if requested
    if args.mmap:
        metadata.collection_schema.mmap_enabled = True

    dump_mgr = LocalDumpManager(settings, metadata)

    # If mmap is requested, we ensure the property is set on the server
    # before the export process triggers self.load_index()
    if args.mmap:
        try:
            # We release first because you can't alter properties on a loaded collection
            dump_mgr.release_index()
        except Exception as e:
            # Defer heavy imports only to the error path to keep startup/autocomplete fast
            from vectordb_operations.milvus_manager import MilvusCollMgrNotFound

            if isinstance(e, MilvusCollMgrNotFound):
                sys.exit(1)
            else:
                pass  # Ignore if not loaded
        dump_mgr.toggle_mmap(True)

    dump_mgr.export_collection_dump(
        root_dump_dir=args.out_dir,
        dump_name=args.dump_name,
    )
    dump_mgr.export_metadata(
        root_dump_dir=args.out_dir,
        dump_name=args.dump_name,
    )
    print("Export complete!")


def run_import(args):
    """Handles the import command."""
    # Defer heavy imports
    from config import load_metadata, load_settings
    from vectordb_operations.milvus_dump import LocalDumpManager

    # Get import files
    dump_dir = args.dump_dir
    metadata_path = args.metadata_path

    # Check if dump_dir exists and is a directory
    if not os.path.isdir(dump_dir):
        raise MilvusDumpCliError(f"Dump directory not found: {dump_dir}")

    # Automatic discovery of metadata.json if not specified
    if not metadata_path:
        print(f"Searching for metadata.json in {dump_dir}...")
        json_files = list(Path(dump_dir).glob("*.json"))
        if not json_files:
            raise MilvusDumpCliError(
                f"No .json metadata file found in {dump_dir}. "
                "Please specify the metadata file with --metadata-path."
            )
        if len(json_files) > 1:
            raise MilvusDumpCliError(
                f"Multiple .json files found in {dump_dir}. "
                "Please specify the metadata file with --metadata-path."
            )
        metadata_path = str(json_files[0])
        print(f"Found metadata: {metadata_path}")

    print(f"Starting import from: {dump_dir}")

    settings = load_settings()
    settings.milvus_host = args.host
    settings.milvus_port = args.port
    settings.minio_host = args.minio_host
    settings.minio_port = args.minio_port

    metadata = load_metadata(metadata_path)

    # Override mmap setting from CLI if provided
    if args.mmap:
        metadata.collection_schema.mmap_enabled = True

    dump_mgr = LocalDumpManager(settings, metadata)

    if args.drop_existing:
        print("Dropping existing collection...")
        dump_mgr.drop_collection()

    print("Creating collection...")
    dump_mgr.create_collection()

    print("Importing data...")
    dump_mgr.import_collection_dump(dump_dir)

    print("Setting up index...")
    dump_mgr.setup_index()
    print("Import complete!")


def run_info_bucket(args):
    """Handles the info bucket command."""
    from config import load_settings, mock_metadata
    from vectordb_operations.milvus_dump import LocalDumpManager

    settings = load_settings()
    settings.milvus_host = args.host
    settings.milvus_port = args.port
    settings.minio_host = args.minio_host
    settings.minio_port = args.minio_port
    if hasattr(args, "bucket") and args.bucket:
        settings.minio_bucket = args.bucket

    metadata = mock_metadata("info")
    dump_mgr = LocalDumpManager(settings, metadata)
    dump_mgr.minio = dump_mgr._get_minio_handler()
    dump_mgr.minio.stats()


def run_info_collections(args):
    """Handles the info collections command."""
    from config import load_settings, mock_metadata
    from vectordb_operations.milvus_manager import MilvusCollectionManager

    settings = load_settings()
    settings.milvus_host = args.host
    settings.milvus_port = args.port
    settings.minio_host = args.minio_host
    settings.minio_port = args.minio_port

    metadata = mock_metadata("info")
    mgr = MilvusCollectionManager(settings, metadata)
    mgr.stats()


def main():
    """Main CLI entry point."""
    settings = load_settings()

    parser = argparse.ArgumentParser(
        description="Milvus Database Dump and Restore Utility",
        formatter_class=CustomHelpFormatter,
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        help="Command to run",
    )

    ###################################
    # EXPORT COMMAND
    export_parser = subparsers.add_parser(
        "export",
        help="Export a Milvus collection",
        formatter_class=CustomHelpFormatter,
    )

    # Add specific arguments first
    export_group = export_parser.add_argument_group("Export Settings")
    export_group.add_argument(
        "-c",
        "--collection",
        required=True,
        help="Name of the collection to export",
    )
    export_group.add_argument(
        "-o",
        "--out-dir",
        default="./milvus_dump",
        help="Root directory for dumps (default: ./milvus_dump)",
    )
    export_group.add_argument(
        "-n",
        "--dump-name",
        default=None,
        help="Custom dump name (default: collection name with timestamp)",
    )

    # Add common arguments after
    add_common_arguments(export_parser, settings)
    export_parser.set_defaults(func=run_export)

    ###################################
    # IMPORT COMMAND
    import_parser = subparsers.add_parser(
        "import",
        help="Import a Milvus collection dump",
        formatter_class=CustomHelpFormatter,
    )

    # Add specific arguments first
    import_group = import_parser.add_argument_group("Import Settings")
    import_group.add_argument(
        "-d",
        "--dump-dir",
        required=True,
        help="Directory containing the Parquet files",
    )
    import_group.add_argument(
        "-m",
        "--metadata-path",
        help="Path to the metadata.json file (optional if inside dump-dir)",
    )
    import_group.add_argument(
        "--drop-existing",
        action="store_true",
        help="WARNING: Drop collection if it already exists",
    )

    # Add common arguments after
    add_common_arguments(import_parser, settings)
    import_parser.set_defaults(func=run_import)

    ###################################
    # INFO COMMAND
    info_parser = subparsers.add_parser(
        "info",
        help="Display information about Milvus collections or MinIO bucket",
        formatter_class=CustomHelpFormatter,
    )
    info_subparsers = info_parser.add_subparsers(
        dest="subcommand",
        required=True,
        help="Information to display",
    )

    # Info Bucket
    bucket_parser = info_subparsers.add_parser(
        "bucket",
        help="Show MinIO bucket statistics",
        formatter_class=CustomHelpFormatter,
    )
    bucket_group = bucket_parser.add_argument_group("Bucket Settings")
    bucket_group.add_argument(
        "-b",
        "--bucket",
        default=settings.minio_bucket,
        help=f"MinIO bucket name (default: {settings.minio_bucket})",
    )
    add_common_arguments(bucket_parser, settings)
    bucket_parser.set_defaults(func=run_info_bucket)

    # Info Collections
    collections_parser = info_subparsers.add_parser(
        "collections",
        help="List all Milvus collections and their row counts",
        formatter_class=CustomHelpFormatter,
    )
    add_common_arguments(collections_parser, settings)
    collections_parser.set_defaults(func=run_info_collections)

    # Enable autocompletion and exclude specific options for safety
    argcomplete.autocomplete(parser, exclude=["--drop-existing"])

    # Parse arguments and run the corresponding function
    args = parser.parse_args()
    try:
        args.func(args)
    except MilvusDumpCliError:
        sys.exit(1)
    except Exception as e:
        # Defer heavy imports only to the error path to keep startup/autocomplete fast
        from vectordb_operations.milvus_manager import (
            MilvusCollMgrError,
            MilvusCollMgrNotFound,
        )
        from vectordb_operations.milvus_dump import LocalDumpMgrError

        if isinstance(
            e,
            (
                MilvusCollMgrError,
                LocalDumpMgrError,
                MilvusCollMgrNotFound,
            ),
        ):
            sys.exit(1)
        else:
            logger.exception(f"An unexpected error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
