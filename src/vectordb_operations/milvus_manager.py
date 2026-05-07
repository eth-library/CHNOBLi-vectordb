import os
import logging
import shutil
from pathlib import Path
from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    MilvusClient,
    FieldSchema,
    DataType,
)
from config import (
    MilvusSettings,
    MilvusCollectionMetadata,
)

from rich.logging import RichHandler
from rich.console import Console

# Create a shared console instance for both logs and progress bars
shared_console = Console()

############################################
# Manager Logger
logger = logging.getLogger(__name__)


class MilvusRichFormatter(logging.Formatter):
    """Custom formatter using Rich markup to preserve your style while being bar-compatible."""

    STYLES = {
        logging.DEBUG: "grey",
        logging.INFO: "white",
        logging.WARNING: "yellow",
        logging.ERROR: "red",
        logging.CRITICAL: "bold red",
    }

    def format(self, record):
        # Determine Style based on log level
        if record.levelno == logging.INFO and record.name.startswith("pymilvus"):
            style = "cyan"
        else:
            style = self.STYLES.get(record.levelno, "white")

        short_name = record.name.split(".")[-1]

        # Determine Format String (identical logic to your current one)
        if record.levelno == logging.INFO:
            if record.name.startswith("pymilvus"):
                msg = f"[{style}] pymilvus | {record.getMessage()}[/]"
            else:
                msg = f"[{style}]{record.getMessage()}[/]"
        else:
            # Warnings, Errors, etc.
            prefix = (
                f"[{record.levelname}][{short_name}]"
                if record.name.startswith("pymilvus")
                else f"[{record.levelname}]"
            )
            msg = f"[{style}]{prefix}: {record.getMessage()}[/]"
            if record.levelno > logging.INFO:
                msg += f" [dim]({record.filename}:{record.lineno})[/]"

        return msg


def setup_clean_logging():
    # Use the shared console to ensure logs don't conflict with the Progress bar
    rich_handler = RichHandler(
        console=shared_console,
        show_time=False,
        show_level=False,  # We handle the level in your custom formatter logic
        show_path=False,
        markup=True,
    )
    rich_handler.setFormatter(MilvusRichFormatter())

    # List of logger prefixes we want to control
    target_loggers = ["pymilvus", "vectordb_operations", "__main__"]

    for name in logging.Logger.manager.loggerDict:
        if any(name.startswith(target) for target in target_loggers):
            target_logger = logging.getLogger(name)
            target_logger.handlers.clear()
            target_logger.propagate = False
            target_logger.addHandler(rich_handler)
            target_logger.setLevel(logging.INFO)


setup_clean_logging()
############################################

# Map config data types to Milvus DataType
TYPE_MAP = {
    "VARCHAR": DataType.VARCHAR,
    "FLOAT_VECTOR": DataType.FLOAT_VECTOR,
    "INT64": DataType.INT64,
}


class MilvusCollMgrError(Exception):
    def __init__(self, message):
        # Call the base class constructor with the parameters it needs
        super().__init__(message)
        # stacklevel=2 makes the logger report the line where the exception was RAISED
        logger.error(message, stacklevel=2)


class MilvusCollMgrNotFound(Exception):
    def __init__(self, message):
        super().__init__(message)
        # stacklevel=2 makes the logger report the line where the exception was RAISED
        logger.error(message, stacklevel=2)


class MilvusCollectionManager:
    def __init__(
        self,
        settings: MilvusSettings,
        metadata: MilvusCollectionMetadata,
    ):
        self.settings = settings
        self.client: MilvusClient | None = None
        self.vector_dim = None

        # Collection name is derived from the modified schema config
        self.schema_config = metadata.collection_schema
        self.index_params = metadata.index
        self.collection_name = self.schema_config.collection_name
        self.collection_description = self.schema_config.description

        # Milvus CollectionSchema object (constructed based on schema_config)
        self.schema: CollectionSchema | None = None

        # Milvus collection object (initialized after schema construction)
        self.collection: Collection | None = None

        # Connection to Milvus is established immediately upon instantiation
        self._connect()

    def _connect(self):
        if not self.client:
            logger.info(
                "Connecting to Milvus at %s:%s ...",
                self.settings.milvus_host,
                self.settings.milvus_port,
            )
            try:
                # Used for direct collection management
                self.client = MilvusClient(
                    uri=self.settings.milvus_endpoint,
                    user=self.settings.milvus_user,
                    password=self.settings.milvus_password,
                    timeout=60,
                )

                # This is REQUIRED for utility.do_bulk_insert to work
                connections.connect(
                    alias="default",
                    host=self.settings.milvus_host,
                    port=self.settings.milvus_port,
                    user=self.settings.milvus_user,
                    password=self.settings.milvus_password,
                )
            except Exception as e:
                raise MilvusCollMgrError(f"Failed to connect to Milvus: {e}")

    def _get_schema(self):
        """
        Constructs a Milvus CollectionSchema based on the provided schema_config.milvus_fields
        """
        if self.schema is not None:
            return self.schema

        # Check if client is connected
        if self.client is None:
            raise MilvusCollMgrError("Milvus client is not connected.")

        # Create a list to hold FieldSchema objects
        fields = []
        self.vector_dim = None

        for field in self.schema_config.fields:
            # Start with the base entries
            field_args = {
                "name": field.name,
                "dtype": field.type,
                "is_primary": field.is_primary,
            }

            # Add optional parameters based on field type
            if field.params.max_length:
                field_args["max_length"] = field.params.max_length
            if field.params.dim:
                field_args["dim"] = field.params.dim
                self.vector_dim = field.params.dim

            # Build the FieldSchema explicitly and append to the list
            field_obj = FieldSchema(**field_args)
            fields.append(field_obj)

        # Create the CollectionSchema
        self.schema = CollectionSchema(
            fields=fields,
            auto_id=False,  # we are providing our own IDs in the dump
            enable_dynamic_field=self.schema_config.enable_dynamic_field,  # allows to add custom fields without breaking the schema
        )

        # We must have a primary field for the bulk writer to work properly
        if self.schema.primary_field is None:
            raise MilvusCollMgrError(
                "Failed to register Primary Field in Milvus Schema!"
            )

        logger.info(
            "Schema initialized. Primary Key: %s",
            self.schema.primary_field.name,
        )

    def _get_collection(self):
        """
        Checks if the collection exists in Milvus. If it does, it initializes
        the collection object.
        """
        if self.collection is not None:
            return self.collection

        if self.client is None:
            raise MilvusCollMgrError("Milvus client is not connected.")

        # Get Collection if it exists
        if self.client.has_collection(self.collection_name):
            logger.info(
                f"Collection '{self.collection_name}' already exists in Milvus."
                " Pulling the official schema..."
            )
            # DO NOT pass the schema. Let Milvus pull the official one.
            self.collection = Collection(name=self.collection_name)
            # CRITICAL: Overwrite your local schema with the official DB schema
            # This guarantees the LocalBulkWriter will not crash!
            self.schema = self.collection.schema
        else:
            raise MilvusCollMgrNotFound(
                f"Collection '{self.collection_name}' does not exist in Milvus."
                "\n\tPlease create the collection first before trying to get it."
            )

    def _move_dump_dir(self, from_dir: str | Path, to_dir: str | Path):
        """
        Move files to a clean directory named after the collection
        """
        # Clean target directory if it exists
        if os.path.exists(to_dir):
            logger.info(f"    -> Cleaning existing dump directory '{to_dir}'...")
            shutil.rmtree(to_dir, ignore_errors=True)
        # Move files
        os.makedirs(to_dir, exist_ok=True)
        for file in os.listdir(from_dir):
            os.rename(os.path.join(from_dir, file), os.path.join(to_dir, file))

    def _clean_temp_dir(self, temp_dir: str | Path):
        """
        Move files to a clean directory named after the collection
        """
        # Clean dump directory if it exists
        if os.path.exists(temp_dir):
            logger.info(f"    -> Cleaning local dump directory '{temp_dir}'...")
            shutil.rmtree(temp_dir, ignore_errors=True)

    def drop_collection(self):
        if self.client is None:
            raise MilvusCollMgrError("Milvus client is not connected.")
        if self.client.has_collection(self.collection_name):
            logger.info(f"Dropping existing collection '{self.collection_name}'...")
            self.client.drop_collection(self.collection_name)

    def create_collection(self):
        """
        Constructs a Milvus CollectionSchema based on the provided schema_config.milvus_fields
        """
        logger.info(f"Creating collection '{self.collection_name}' in Milvus...")
        # Check if client is connected
        if self.client is None:
            raise MilvusCollMgrError("Milvus client is not connected.")
        # Get or construct the schema
        if self.schema is None:
            self._get_schema()

        # Create the collection in Milvus using the constructed schema
        if self.client.has_collection(self.collection_name):
            raise MilvusCollMgrError(
                f"Collection '{self.collection_name}' already exists in Milvus. Cannot create a new collection with the same name."
                "\n\tPlease drop the existing collection first or choose a different collection name in the schema config and try again."
            )

        # Set properties like mmap
        properties = {}
        if self.schema_config.mmap_enabled:
            logger.info("    -> Enabling mmap for this collection...")
            properties["mmap.enabled"] = "true"

        self.client.create_collection(
            collection_name=self.collection_name,
            schema=self.schema,
            properties=properties,
        )
        self._get_collection()

    def toggle_mmap(self, enabled: bool):
        """
        Enables or disables mmap for the collection.
        Note: You must release the collection from memory before changing this property.
        """
        if self.client is None:
            raise MilvusCollMgrError("Milvus client is not connected.")

        logger.info(
            f"Setting mmap.enabled={enabled} for collection '{self.collection_name}'..."
        )
        self.client.alter_collection_properties(
            collection_name=self.collection_name,
            properties={"mmap.enabled": str(enabled).lower()},
        )

    def _is_compatible_index(self, index_description):
        target_params = self.index_params

        # Extract actual values from Milvus
        actual_type = index_description.get("index_type")
        actual_metric = index_description.get("metric_type")
        # Handle cases where nlist might not be present in the index description
        actual_nlist = index_description.get("nlist")
        if actual_nlist is not None:
            actual_nlist = int(actual_nlist)

        # Compare basic types
        type_match = actual_type == target_params.index_type
        metric_match = actual_metric == target_params.metric_type
        # nlist match: both are None, or they are equal
        nlist_match = actual_nlist == target_params.params.nlist

        if type_match and metric_match and nlist_match:
            return True
        else:
            logger.warning(
                f"Found index '%s' on collection '%s'",
                index_description.get("index_name", ""),
                self.collection_name,
            )
            logger.warning(f"Index compare: ( field: schema | remote_index )")
            for (name_type, target_type, actual_type), _ in zip(
                [
                    ("index", target_params.index_type, actual_type),
                    ("metric", target_params.metric_type, actual_metric),
                    ("nlist", target_params.params.nlist, actual_nlist),
                ],
                [type_match, metric_match, nlist_match],
            ):
                logger.warning(f" - {name_type}: {target_type} | {actual_type}")

            return False

    def drop_index(self, index_name):
        """
        Drops the index on the specified field if it exists.
        This is useful for testing index creation and ensuring a clean state.
        """
        if self.client is None:
            raise MilvusCollMgrError("Build Index: Milvus client is not connected.")
        # Only drop if there's actually an index to drop
        indices = self.client.list_indexes(self.collection_name)
        if indices:
            if index_name in indices:
                self.client.drop_index(self.collection_name, index_name=index_name)
            else:
                logger.warning(
                    "No index named '%s' found on collection '%s'. Skipping drop.",
                    index_name,
                    self.collection_name,
                )
                logger.warning(f"Existing indices: {indices}\n")

    def setup_index(self, use_existing=False):
        """
        Builds the index on the specified field using the parameters defined in the schema config.
        This should be called after loading data to ensure the index is built on the newly inserted vectors.
        """
        self._get_collection()

        if self.collection is None:
            raise MilvusCollMgrError(
                "Build Indes: Milvus collection is not initialized."
            )
        if self.client is None:
            raise MilvusCollMgrError("Build Index: Milvus client is not connected.")

        # Check for existing indexes
        index_description = self.client.describe_index(
            collection_name=self.collection_name, index_name=""
        )
        # If an index exists, check compatibility with current schema config
        if not use_existing and index_description:
            if self._is_compatible_index(index_description):
                raise MilvusCollMgrError(
                    f"\n\tCompatible index already exists for collection '{self.collection_name}'."
                    "\n\tIf you want to use the existing index, set use_existing=True in setup_index and run again."
                )
            else:
                raise MilvusCollMgrError(
                    f"\n\tExisting index on '{self.collection_name}' is NOT compatible with the current schema config."
                    "\n\tIf you want to use the existing index, set use_existing=True in setup_index and run again."
                    "\n\tOtherwise, please drop the index manually in Milvus and run this again to create a new one with the correct parameters."
                    "\n\t(dropping the index will not delete your data)"
                )

        # If use_existing is True but no index is found, raise an error to avoid silent failures
        if use_existing and index_description is None:
            raise MilvusCollMgrError(
                "\n\tSetup Index: cannot use existing index because no index was found in Milvus."
                "\n\tPlease create the index first or set use_existing=False to create a new one."
            )

        if use_existing:
            return

        # Get the vector field name dynamically from your fields list
        vector_field = next(f.name for f in self.schema_config.fields if f.params.dim)

        # Prepare index parameters from config
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name=vector_field,
            index_type=self.index_params.index_type,
            metric_type=self.index_params.metric_type,
            params=self.index_params.params.model_dump(exclude_none=True),
        )

        # Create the index
        logger.info("    -> Creating index...")
        self.client.create_index(
            collection_name=self.collection_name,
            index_params=index_params,
        )
        logger.info("    -> Indexing completed")

    def load_index(self):
        """
        Loads the collection index into memory.
        This is a separate step in Milvus that can be done after index creation and data insertion.
        """
        self._connect()
        self._get_collection()

        if self.collection is None:
            raise MilvusCollMgrError("Exporter: Collection is not initialized.")
        if self.client is None:
            raise MilvusCollMgrError("Exporter: Milvus client is not connected.")

        logger.info("    -> Loading collection index into memory...")
        # Check if there is an index
        if self.client.list_indexes(self.collection_name) == []:
            raise MilvusCollMgrError(
                "Load Index: No index found on the collection. Create an index before loading."
            )
        self.client.load_collection(collection_name=self.collection_name)

    def release_index(self):
        """
        Releases the collection index from memory.
        """
        self._connect()
        self._get_collection()

        if self.collection is None:
            raise MilvusCollMgrError("Exporter: Collection is not initialized.")
        if self.client is None:
            raise MilvusCollMgrError("Exporter: Milvus client is not connected.")

        logger.info("    -> Releasing collection index from memory...")
        # Check if there is an index
        if self.client.list_indexes(self.collection_name) == []:
            raise MilvusCollMgrError(
                "Load Index: No index found on the collection. Create an index before loading."
            )
        self.client.release_collection(collection_name=self.collection_name)

    def list_collections_info(self) -> list[dict]:
        """
        Returns a list of all collections with their row counts.
        """
        if self.client is None:
            raise MilvusCollMgrError("Milvus client is not connected.")

        collections = self.client.list_collections()
        if not collections:
            logger.warning("No collections found in Milvus.")
            return []

        info = []
        for name in collections:  # type:ignore
            try:
                stats = self.client.get_collection_stats(collection_name=name)
                info.append({"name": name, "row_count": stats.get("row_count", 0)})
            except Exception as e:
                logger.warning(f"Could not get stats for collection '{name}': {e}")
                info.append({"name": name, "row_count": "Unknown"})
        return info

    def stats(self):
        """
        Print out all the collections and their size
        """
        from rich.table import Table

        self._connect()

        if self.client is None:
            raise MilvusCollMgrError("Milvus client is not connected.")

        collections_info = self.list_collections_info()
        table = Table(
            title="Milvus Collections", show_header=True, header_style="bold magenta"
        )
        table.add_column("Collection Name", style="cyan", no_wrap=True)
        table.add_column("Row Count", justify="right", style="green")
        for row in sorted(collections_info, key=lambda x: x["name"]):
            table.add_row(row["name"], f"{row["row_count"]:,}")
        shared_console.print(table)


if __name__ == "__main__":
    from config import load_settings, load_example_metadata

    # Use it before you upload
    metadata = load_example_metadata()
    settings = load_settings()

    # Prepare milvus connection
    mgr = MilvusCollectionManager(settings, metadata)

    mgr.stats()
