import json
from pathlib import Path
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Get the absolute paths or root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Get path to env and schema.json
ENV_PATH = PROJECT_ROOT.joinpath(".env")
ENV_EXAMPLE_PATH = PROJECT_ROOT.joinpath("config/env.example")
SCHEMA_PATH = PROJECT_ROOT.joinpath("config", "example_metadata.json")


# ==========================================
# ENVIRONMENT VARIABLES VALIDATION
# ==========================================
class MilvusSettings(BaseSettings):
    """
    Pydantic automatically reads from the .env file.
    If any of these are missing or the wrong type, the app crashes immediately with a clear error.
    """

    # Milvus Settings
    milvus_host: str = "localhost"
    milvus_port: int = 19530  # Defaults to 19530 if not provided
    milvus_user: str = ""
    milvus_password: str = ""

    # MinIO/S3 Settings
    minio_host: str = "localhost"
    minio_port: int = 9000
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket: str = "a-bucket"

    @property
    def milvus_endpoint(self) -> str:
        return f"http://{self.milvus_host}:{self.milvus_port}"

    @property
    def minio_endpoint(self) -> str:
        return f"http://{self.minio_host}:{self.minio_port}"

    model_config = SettingsConfigDict(
        env_file=(ENV_EXAMPLE_PATH, ENV_PATH),
        env_file_encoding="utf-8",
        extra="ignore",
    )


# ==========================================
# SCHEMA.JSON VALIDATION
# ==========================================
class FieldParams(BaseModel):
    """Handles the nested 'params' dictionary."""

    max_length: int | None = None
    dim: int | None = None


class MilvusField(BaseModel):
    name: str
    description: str = ""
    type: int
    is_primary: bool = False
    params: FieldParams


class IndexParams(BaseModel):
    nlist: int | None = Field(None, gt=0, description="Must be a positive integer")


class MilvusIndex(BaseModel):
    index_type: str
    metric_type: str
    params: IndexParams = Field(default_factory=IndexParams)  # type: ignore

    @model_validator(mode="before")
    @classmethod
    def wrap_params(cls, data):
        if isinstance(data, dict):
            # If 'params' is missing but 'nlist' is present at root, wrap it
            if "params" not in data and "nlist" in data:
                data["params"] = {"nlist": data.pop("nlist")}
        return data


class MilvusSchema(BaseModel):
    """
    Validates the entire schema.json file structure.
    """

    collection_name: str
    aliases: list[str]
    num_shards: int = Field(gt=0, description="Must be a positive integer")
    description: str = ""
    fields: list[MilvusField]
    auto_id: bool = False
    enable_dynamic_field: bool = False
    mmap_enabled: bool = False


class MilvusCollectionMetadata(BaseModel):
    """
    Validates the metadata structure for exported dumps.
    """

    collection_name: str
    collection_schema: MilvusSchema = Field(..., alias="schema")
    index: MilvusIndex

    @model_validator(mode="after")
    def validate_collection_names_match(self) -> "MilvusCollectionMetadata":
        if self.collection_name != self.collection_schema.collection_name:
            raise ValueError(
                f"Collection name '{self.collection_name}' does not match "
                f"schema collection name '{self.collection_schema.collection_name}'"
            )
        return self


# ==========================================
# LOADERS
# ==========================================
def load_settings() -> MilvusSettings:
    return MilvusSettings()


def load_example_metadata() -> MilvusCollectionMetadata:
    with open(SCHEMA_PATH, "r") as f:
        data = json.load(f)
    # Pydantic validates the dictionary here!
    return MilvusCollectionMetadata(**data)


def load_metadata(path: str) -> MilvusCollectionMetadata:
    with open(path, "r") as f:
        data = json.load(f)
    # Pydantic validates the dictionary here!
    return MilvusCollectionMetadata(**data)


def mock_metadata(collection_name: str) -> MilvusCollectionMetadata:
    """
    Creates a mock metadata object based on the example metadata, but with a custom collection name.
    """
    loaded = load_example_metadata()
    loaded.collection_name = collection_name
    loaded.collection_schema.collection_name = collection_name
    return loaded
