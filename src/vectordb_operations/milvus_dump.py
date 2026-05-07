import os
import time
import logging
import shutil
import json
import numpy as np
import boto3
from botocore.exceptions import ClientError
from pathlib import Path
from pymilvus import utility
from pymilvus.bulk_writer import LocalBulkWriter, BulkFileType
from config import (
    load_settings,
    load_example_metadata,
    MilvusSettings,
    MilvusCollectionMetadata,
)
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
)
from vectordb_operations.milvus_manager import MilvusCollectionManager, shared_console

# Get logger for this module
logger = logging.getLogger(__name__)

# Set for the number of rows to fetch per batch when exporting data from Milvus
BATCH_SIZE = 500


class bcolors:
    DEBUG = "\033[0;2m"  # Grey
    OK = "\033[32m"  # Green
    INFO = "\033[34m"  # Cyan
    WARNING = "\033[33m"  # Yellow
    ERROR = "\033[31m"  # Red
    CRITICAL = "\033[1;31m"  # Bold Red
    RESET = "\033[0m"
    BOLD = "\033[1m"


class S3StorageHandler:
    def __init__(self, endpoint, access_key: str, secret_key: str, bucket: str):
        self.s3_client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )
        self.bucket = bucket

    def upload_directory(
        self,
        s3_prefix: str,
        dump_data_dir: str,
    ):
        """
        Uploads Parquet files from a local directory to MinIO under a specified S3 prefix.
        Args:
            s3_prefix (str): The S3 prefix (folder path) under which the files will
                                be uploaded (e.g., "milvus_imports/collection_name/")
            dump_data_dir (str): The local directory containing the Parquet files to be uploaded.
        """
        logger.info("Uploading Parquet files to MinIO...")
        # Ensure the bucket exists
        try:
            self.s3_client.head_bucket(Bucket=self.bucket)
        except ClientError:
            self.s3_client.create_bucket(Bucket=self.bucket)

        # Clean up any existing files in the target S3 prefix
        self.clear_s3_prefix(s3_prefix)

        # Find the parquet files in the dump directory
        uploaded_s3_paths = []
        for file in os.listdir(dump_data_dir):
            if file.endswith(".parquet"):
                local_filepath = os.path.join(dump_data_dir, file)
                # Create a clean path for S3 (e.g., "dummy_imports/1.parquet")
                # s3_key = f"{s3_prefix}{file}"
                s3_key = os.path.join(s3_prefix, file)
                logger.info(
                    f"    -> Uploading {local_filepath} to s3://{self.bucket}/{s3_key}..."
                )
                self.s3_client.upload_file(local_filepath, self.bucket, s3_key)
                uploaded_s3_paths.append(s3_key)

        return uploaded_s3_paths

    def clear_s3_prefix(self, prefix):
        """
        Deletes everything under a specific 'folder' in S3/MinIO.
        Uses singular delete_object to avoid 'MissingContentMD5' errors on MinIO.
        """
        # List all objects in the prefix
        paginator = self.s3_client.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=self.bucket, Prefix=prefix)

        deleted_count = 0
        try:
            for page in pages:
                if "Contents" in page:
                    for obj in page["Contents"]:
                        # Use delete_object
                        self.s3_client.delete_object(Bucket=self.bucket, Key=obj["Key"])
                        deleted_count += 1

            if deleted_count > 0:
                logger.info(
                    f"    -> Purged {deleted_count} existing files in s3://{self.bucket}/{prefix}"
                )
            else:
                logger.info(f"    -> {prefix} is already empty.")

        except Exception as e:
            logger.info(f"    -> Warning: Cleanup failed (but continuing anyway): {e}")

    def list_bucket_info(self):
        """
        Lists all objects in bucket and the content in MinIO for debugging purposes.
        """
        info = []
        try:
            objects = self.s3_client.list_objects_v2(Bucket=self.bucket)
            for content in objects.get("Contents", []):
                if content["Key"].startswith("milvus_imports/"):
                    info.append(
                        {
                            "name": content["Key"],
                            "size": content["Size"],
                            "timestamp": content["LastModified"].strftime(
                                "%Y-%m-%d %H:%M:%S"
                            ),
                        }
                    )

        except Exception as e:
            logger.error(f"Failed to list bucket contents: {e}")
        return info

    def stats(self):
        """
        Prints stats about the bucket and its contents for debugging purposes.
        """
        from rich.table import Table

        info = self.list_bucket_info()
        table = Table(title=f"Contents of Bucket: {self.bucket}")
        table.add_column("Name", justify="left", style="cyan")
        table.add_column("Size", justify="right", style="magenta")
        table.add_column("Last Modified", justify="center", style="green")
        for item in info:
            size = item["size"]
            units = ["KB", "MB", "GB", "TB"]
            unit = "Bytes"
            while size > 1000:
                size /= 1000
                unit = units.pop(0)

            table.add_row(item["name"], f"{size:.2f}{unit}", item["timestamp"])
        shared_console.print(table)


class LocalDumpMgrError(Exception):
    """Custom exception for LocalDumpManager errors."""

    def __init__(self, message):
        super().__init__(message)
        # stacklevel=2 makes the logger report the line where the exception was RAISED
        logger.error(message, stacklevel=2)


class LocalDumpManager(MilvusCollectionManager):
    def __init__(
        self,
        settings: MilvusSettings,
        metadata: MilvusCollectionMetadata,
    ):
        super().__init__(settings, metadata)
        self.metadata = metadata
        # Initialize MinIO handler as None
        self.minio: S3StorageHandler | None = None

    def _get_minio_handler(self) -> S3StorageHandler:
        if self.minio is None:
            return S3StorageHandler(
                endpoint=self.settings.minio_endpoint,
                access_key=self.settings.minio_access_key,
                secret_key=self.settings.minio_secret_key,
                bucket=self.settings.minio_bucket,
            )
        else:
            return self.minio

    # def set_root_dump_dir(self, root_dump_dir=DUMP_DIR):
    #    """Sets the root directory for dump input/output. This allows you to specify a custom location for your dump files."""
    #    self.root_dump_dir = root_dump_dir

    def _export_dir(self, root_dump_dir="./milvus_dump") -> str:
        """Appends a date tag to the dump directory for better organization of multiple dumps over time.
        The format of the date tag is YYYYMMDD
        """
        date_tag = time.strftime("%Y%m%d")
        return f"{root_dump_dir}/{self.collection_name}_{date_tag}"

    def create_dummy_dump(
        self,
        root_dump_dir="./milvus_dump",
        dump_name=None,
        size=10_000,
        segment_size_mb=512,
    ) -> str | Path:
        """
        Writes dummy data to local Parquet files using LocalBulkWriter.
        The data is generated based on the schema configuration.
        """
        self._get_schema()

        # Generate random vectors based on the vector dimension defined in the schema
        if self.vector_dim is None:
            raise LocalDumpMgrError(
                "Vector dimension is not defined in the schema. Cannot generate dummy data."
            )
        # Get or construct the schema
        if self.schema is None:
            raise LocalDumpMgrError("Dummy Creator: Failed to construct schema.")

        # print("\n" + "=" * 50)
        logger.info(
            "Generating %d random vectors (Dimension: %d)...",
            size,
            self.vector_dim,
        )
        random_vectors = np.random.rand(size, self.vector_dim).astype(np.float32)

        logger.info("    -> Writing data to local Parquet files via LocalBulkWriter...")
        writer = LocalBulkWriter(
            schema=self.schema,
            local_path=root_dump_dir,
            segment_size=segment_size_mb
            * 1024
            * 1024,  # 512MB segment size for testing
            file_type=BulkFileType.PARQUET,
        )

        for i in range(size):
            writer.append_row(
                {
                    "text_id": f"dummy_bulk_{i:011_d}",
                    "embedding": random_vectors[i].tolist(),
                }
            )

        # Finalize the writer to ensure all data is flushed to disk
        writer.commit()
        dump_data_dir = writer.data_path

        # Move files to a clean directory named after the collection
        final_dump_dir = self._export_dir(root_dump_dir)
        if dump_name is not None:
            final_dump_dir = os.path.join(root_dump_dir, dump_name)

        self._move_dump_dir(from_dir=dump_data_dir, to_dir=final_dump_dir)

        logger.info(
            "    -> Dump successfully created in '%s'",
            os.path.abspath(final_dump_dir),
        )
        return os.path.abspath(final_dump_dir)

    def import_collection_dump(self, dump_data_dir):
        """
        Loads Parquet files from the specified local directory into Milvus
        using the bulk insert utility.
        """
        # Check if client is connected
        if self.client is None:
            raise LocalDumpMgrError("Milvus client is not connected.")

        # Ensure MinIO connection is established
        self.minio = self._get_minio_handler()

        # Create s3 prefix for this collection's dumps
        s3_prefix = f"milvus_imports/{self.schema_config.collection_name}/"
        uploaded_s3_paths = self.minio.upload_directory(s3_prefix, dump_data_dir)

        # Container for tracking background task IDs and total inserted rows
        active_tasks = []
        total_inserted_rows = 0

        # Start the ingestion stopwatch
        ingestion_start_time = time.time()

        # Submit a separate task for each Parquet file
        for s3_path in uploaded_s3_paths:
            task_id = utility.do_bulk_insert(
                collection_name=self.collection_name, files=[s3_path]
            )
            active_tasks.append({"milvus_id": task_id, "file": s3_path})
            logger.info(f"    -> Submitted task for {s3_path} (Task ID: {task_id})")

        logger.info("    -> Waiting for all background tasks to complete...")

        # Monitor all tasks until the list is empty
        while active_tasks:
            for task in active_tasks[:]:
                state = utility.get_bulk_insert_state(task_id=task["milvus_id"])

                # TODO: we can use state.state_name to update the user on the progress
                # of each file (e.g., "Processing", "Completed", "Failed", etc.)
                if state.state_name == "Completed":
                    logger.info(
                        f"    -> [Completed] Task %d loaded %d rows.",
                        task["milvus_id"],
                        state.row_count,
                    )
                    total_inserted_rows += state.row_count
                    active_tasks.remove(task)

                elif state.state_name in ["Failed", "FailedAndCleaned"]:
                    logger.error(
                        f"    -> [FAILED] Task %d. Reason: %s",
                        task["milvus_id"],
                        state.infos.get("failed_reason"),
                    )
                    self.drop_collection()
                    exit(1)

            if active_tasks:
                # Wait 3 seconds before checking status again
                time.sleep(3)

        # Print Stats of insertion
        ingestion_end_time = time.time()
        elapsed_seconds = ingestion_end_time - ingestion_start_time
        elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_seconds))
        rows_per_second = (
            total_inserted_rows / elapsed_seconds if elapsed_seconds > 0 else 0
        )
        logger.info("=" * 50)
        logger.info(f"Ingestion Benchmark Results:")
        logger.info(f" - Total rows loaded: {total_inserted_rows}")
        logger.info(f" - Total time:        {elapsed_time}")
        logger.info(f" - Ingestion speed:   {rows_per_second:.0f} rows / second")
        logger.info("=" * 50)

        # Clean the import files in the target S3 prefix bucket
        print("\nCleaning up import files in MinIO...")
        self.minio.clear_s3_prefix(s3_prefix)

    def export_collection_dump(
        self,
        root_dump_dir="./milvus_dump",
        dump_name=None,
        segment_size_mb=512,
    ) -> Path | str:
        """
        Create export files based on a target file size (e.g., 512MB) rather than a fixed number of rows.
        """
        self._connect()
        self._get_collection()

        # We must rebuild the Milvus Schema object from your Pydantic config
        if self.schema is None:
            raise LocalDumpMgrError("Exporter: Schema config is not defined.")
        if self.collection is None:
            raise LocalDumpMgrError("Exporter: Collection is not initialized.")
        if self.client is None:
            raise LocalDumpMgrError("Exporter: Milvus client is not connected.")

        # Create output directory if it doesn't exist
        os.makedirs(root_dump_dir, exist_ok=True)

        # Define the final dump directory name (e.g., root_dump_dir/collection_name_dateTag)
        final_dump_dir = self._export_dir(root_dump_dir)
        if dump_name is not None:
            final_dump_dir = os.path.join(root_dump_dir, dump_name)

        # Check if output_dir already exists
        if os.path.exists(final_dump_dir):
            raise LocalDumpMgrError(
                f"Exporter: Output directory '{final_dump_dir}' already exists. Please choose a different name or remove it before exporting."
            )

        # Load index if it exists
        self.load_index()

        total_rows = self.collection.num_entities

        logger.info(
            f"Starting export of {total_rows:,} rows (Target: %dMB per file)...",
            segment_size_mb,
        )
        # Setup the LocalBulkWriter to write files to a temporary directory
        writer = LocalBulkWriter(
            schema=self.schema,
            local_path=root_dump_dir,
            segment_size=segment_size_mb * 1024 * 1024,
            file_type=BulkFileType.PARQUET,
        )

        # Pull data via Iterator
        iterator = self.client.query_iterator(
            collection_name=self.collection_name,
            batch_size=BATCH_SIZE,
            output_fields=["*"],
        )

        # Flag if export was successful, used for cleanup in case of failure
        export_successful = False

        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                MofNCompleteColumn(),
                TimeRemainingColumn(),
                console=shared_console,
                transient=True,
            ) as progress:
                export_task = progress.add_task(
                    "[cyan]Exporting rows...", total=total_rows
                )

                while True:
                    batch = iterator.next()
                    if not batch:
                        break

                    for row in batch:
                        # writer.append_row will automatically trigger a file split
                        # whenever the cumulative size hits your segment_size_mb limit.
                        writer.append_row(row)

                    progress.update(export_task, advance=len(batch))

            writer.commit()
            dump_data_dir = writer.data_path
            # Move files to a clean directory named after the collection
            self._move_dump_dir(from_dir=dump_data_dir, to_dir=final_dump_dir)
            export_successful = True

        finally:
            # If export was not successful, attempt to clean up any temporary files and release resources
            if not export_successful:
                # If the writer is currently doing something, try to let it finish
                # its current background thread so it releases file locks on the OS.
                if writer is not None:
                    try:
                        # We commit to empty the buffers, knowing we are about to delete it anyway
                        writer.commit()
                    except:
                        pass  # Ignore if the writer itself crashed
                logger.error("Export Failed. Cleaning up temporary files...")
                self._clean_temp_dir(writer.data_path)
                self.release_index()
                del writer  # Ensure writer is deleted to release any file locks
                exit(1)

            iterator.close()

        logger.info(f"    -> Export finished. Files saved in: {final_dump_dir}")

        # Release index from memory after export
        self.release_index()
        return final_dump_dir

    def export_metadata(
        self,
        root_dump_dir="./milvus_dump",
        dump_name=None,
    ):
        """
        Queries the Milvus server for the live collection schema and index info,
        and saves it to a JSON file.
        """
        self._get_collection()

        if self.client is None:
            raise LocalDumpMgrError("Exporter: Milvus client is not connected.")
        if self.collection is None:
            raise LocalDumpMgrError("Exporter: Collection is not initialized.")

        # Dump directory
        dump_dir = self._export_dir(root_dump_dir)
        if dump_name is not None:
            dump_dir = os.path.join(root_dump_dir, dump_name)
        os.makedirs(dump_dir, exist_ok=True)

        # output file path
        file_path = os.path.join(dump_dir, f"{self.collection_name}_metadata.json")

        # Get the Schema
        collection_info = self.client.describe_collection(self.collection_name)
        # Get the Live Index Info
        index_info = self.client.describe_index(self.collection_name, index_name="")

        # Combine them into a single dictionary
        live_state = {
            "collection_name": self.collection_name,
            "schema": collection_info,
            "index": index_info,
        }

        # Save to file
        with open(file_path, "w") as f:
            json.dump(live_state, f, indent=4)

        logger.info(f"Live Server Metadata saved to: {file_path}")

    def clean_dump(self, dump_dir: str | Path):
        """
        Cleans up local dump directory
        """

        # prompt user for confirmation before proceeding with cleanup
        print(
            f"\nCleanup Utility: This will {bcolors.WARNING}PERMANENTLY DELETE{bcolors.RESET} data"
        )
        print(f"{bcolors.BOLD}Data included in cleanup:{bcolors.RESET}")
        print(f" - Local dump directory:")
        print(f"\t -{bcolors.ERROR}'{dump_dir}'{bcolors.RESET}")

        confirm = input("Are you sure you want to proceed? (yes/[no]): ")
        if confirm.lower() != "yes":
            print("Cleanup aborted by user.")
            return

        # Clean up local dump directory
        print("Cleaning up local dump directories...")
        self._clean_temp_dir(dump_dir)

    def clean_bucket(self):
        """
        Cleans up  MinIO S3 prefixes related to the collection.
        """
        # s3_prefix for the collection's dumps in MinIO
        s3_prefix = f"milvus_imports/{self.schema_config.collection_name}/"

        # prompt user for confirmation before proceeding with cleanup
        print(
            f"\nCleanup Utility: This will {bcolors.WARNING}PERMANENTLY DELETE{bcolors.RESET} data"
        )
        print(f"{bcolors.BOLD}Data included in cleanup:{bcolors.RESET}")
        print(" - buckets/prefixes in MinIO starting with:")
        print(
            f"\t - {bcolors.ERROR}'milvus_imports/{self.schema_config.collection_name}/'{bcolors.RESET}"
        )
        confirm = input("Are you sure you want to proceed? (yes/[no]): ")
        if confirm.lower() != "yes":
            print("Cleanup aborted by user.")
            return

        # connect to MinIO connection
        self.minio = self._get_minio_handler()

        # Clean up any existing files in the target S3 prefix
        self.minio.clear_s3_prefix(s3_prefix)


if __name__ == "__main__":
    # Use it before you upload
    metadata = load_example_metadata()
    settings = load_settings()

    # # Prepare milvus connection
    dump_mgr = LocalDumpManager(settings, metadata)

    ## # # Clean build
    ## dump_mgr.drop_collection()
    ## dump_mgr.create_collection()

    # Get stats of the buckets
    dump_mgr.minio = dump_mgr._get_minio_handler()
    dump_mgr.minio.stats()
    # print(dump_mgr.minio.list_buckets_info())

    # Create dummy data dump
    # dump_dir = dump_mgr.create_dummy_dump(size=10_000)

    ## # Upload to MinIO and import to Milvus
    ## dump_mgr.import_collection_dump(dump_data_dir=dump_dir)
    ## dump_mgr.setup_index()

    # Export collection dump
    # default output_dir is : DUMP_DIR/collection_name_dateTag
    # milvus.export_collection_dump()

    # Clean up
    # dump_mgr.clean_dump(dump_dir=dump_dir)
