import pytest
import boto3
from botocore.exceptions import EndpointConnectionError, ClientError
from pymilvus import connections, utility
from config import load_settings


@pytest.fixture(scope="module")
def settings():
    """Loads the environment variables via Pydantic before tests run."""
    return load_settings()


def test_milvus_connection(settings):
    """Verifies that we can connect to Milvus and ping the server."""
    alias = "pytest_milvus"

    try:
        # Attempt connection
        connections.connect(
            alias=alias,
            host=settings.milvus_host,
            port=settings.milvus_port,
            user=settings.milvus_user,
            password=settings.milvus_password,
        )

        # Assert the connection is registered in the Python client
        assert connections.has_connection(alias) == True

        # Actually ping the server to ensure it is responsive and credentials are valid
        server_version = utility.get_server_version(using=alias)
        assert server_version is not None
        print(f"\nMilvus connection successful! Version: {server_version}")

    finally:
        # Always clean up the connection, even if the test fails
        connections.disconnect(alias)


def test_minio_connection(settings):
    # Attempt connection
    s3_client = boto3.client(
        "s3",
        endpoint_url=settings.minio_endpoint,
        aws_access_key_id=settings.minio_access_key,
        aws_secret_access_key=settings.minio_secret_key,
    )
    try:
        # list_buckets(): try call
        s3_client.list_buckets()
    except EndpointConnectionError:
        pytest.fail("MinIO cannot be reached. Check endpoint and port.")
    except ClientError as e:
        pytest.fail(f"Failed authentication: {e}")
    except Exception as e:
        pytest.fail(f"Failed Connection: {e}")

    finally:
        s3_client.close()
