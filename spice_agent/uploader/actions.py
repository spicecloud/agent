import hashlib
from pathlib import Path
import sys
import threading
from typing import Dict, Optional

import boto3
from boto3.s3.transfer import TransferConfig
import botocore.exceptions
from gql import gql

from spice_agent.graphql.sdk import create_requests_session


class ProgressPercentage(object):
    def __init__(self, filepath: Path) -> None:
        self._filename = filepath.name
        self._size = float(filepath.stat().st_size)
        self._seen_so_far = 0
        self._lock = threading.Lock()

    def __call__(self, bytes_amount):
        # To simplify, assume this is hooked up to a single filename
        with self._lock:
            self._seen_so_far += bytes_amount
            percentage = (self._seen_so_far / self._size) * 100
            sys.stdout.write(
                "\r%s  %s / %s  (%.2f%%)"
                % (self._filename, self._seen_so_far, self._size, percentage)
            )
            sys.stdout.flush()


class Uploader:
    def __init__(self, spice) -> None:
        self.spice = spice

    def _get_access(self):
        query = gql(
            """
            query getAgentUploaderAccess {
                getAgentUploaderAccess {
                    accessKeyId
                    secretAccessKey
                }
            }
        """  # noqa
        )
        result = self.spice.session.execute(query)
        return {
            "access_key_id": result["getAgentUploaderAccess"]["accessKeyId"],
            "secret_access_key": result["getAgentUploaderAccess"]["secretAccessKey"],
        }

    def _create_file(
        self, file_name: str, file_size: int, file_checksum: str, location: str
    ):
        mutation = gql(
            """
            mutation createFile($input: CreateFileInput!) {
                createFile(input: $input) {
                    ... on File {
                        id
                    }
                }
            }
        """  # noqa
        )
        input = {
            "fileName": file_name,
            "fileSize": file_size,
            "fileChecksum": file_checksum,
            "location": location,
        }
        variables = {"input": input}
        result = self.spice.session.execute(mutation, variable_values=variables)
        return result.get("createFile").get("id")

    def _update_file_status(
        self,
        file_id: str,
        is_uploading: Optional[bool] = None,
        is_complete: Optional[bool] = None,
    ):
        mutation = gql(
            """
            mutation updateFileStatus($input: UpdateFileStatusInput!) {
                updateFileStatus(input: $input) {
                    ... on File {
                        id
                    }
                }
            }
        """  # noqa
        )
        input: Dict[str, str | bool] = {
            "fileId": file_id,
        }
        if is_uploading is not None:
            input["isUploading"] = is_uploading
        if is_complete is not None:
            input["isComplete"] = is_complete

        variables = {"input": input}
        result = self.spice.session.execute(mutation, variable_values=variables)
        return result

    def upload_file_direct(
        self,
        bucket_name: str,
        key: str,
        filepath: Path,
        file_id: Optional[str],
        overwrite: bool = False,
    ):
        """
        bucket: name of S3 bucket ie spice-models
        key: dirs and file inside bucket /some/dir/file.json
        """
        if filepath.exists() is False:
            raise Exception(f"File {filepath} does not exist.")

        file_size = filepath.stat().st_size
        if file_size == 0:
            raise Exception(f"Model path {filepath} is empty.")

        file_checksum = hashlib.md5(filepath.read_bytes()).hexdigest()

        if not file_id:
            file_id = self._create_file(
                file_name=filepath.name,
                file_size=filepath.stat().st_size,
                file_checksum=file_checksum,
                location=f"s3://{key}",
            )

        if not file_id:
            raise Exception("No file_id found or provided.")

        s3_access_values = self._get_access()
        s3_client = boto3.client(
            "s3",
            aws_access_key_id=s3_access_values["access_key_id"],
            aws_secret_access_key=s3_access_values["secret_access_key"],
        )

        self._update_file_status(file_id=file_id, is_uploading=True, is_complete=False)

        file_already_uploaded = False
        try:
            head_object = s3_client.head_object(Bucket=bucket_name, Key=key)

            # TODO: add a check here that the checksum is the same as well
            if (
                head_object["ResponseMetadata"]["HTTPHeaders"]["content-length"]
                == file_size
            ):
                file_already_uploaded = True
                print("File already uploaded.")
            else:
                print("File in bucket has a file size mismatch. Reuploading.")
        except botocore.exceptions.ClientError as exception:
            if exception.response["Error"]["Code"] in ["404", "403"]:
                # The key does not exist and we should upload the file
                pass
            # elif exception.response["Error"]["Code"] == "403":
            #     # Unauthorized, including invalid bucket
            #     # raise exception
            #     raise Exception("Unauthorized to view bucket.")
            else:
                print("Unhandled exception.")
                raise exception

        if (file_already_uploaded is False) or (
            file_already_uploaded and overwrite is True
        ):
            # TODO: configure these values based on the available system properties
            # if a file is bigger than multipart_threshold, then do multipart upload
            # multipart_threshold = 1024 * 100
            # multipart_chunksize = 1024 * 100
            max_concurrency = 16
            config = TransferConfig(
                # multipart_threshold=multipart_threshold,
                max_concurrency=max_concurrency,
                # multipart_chunksize=multipart_chunksize,  # 25MB
                use_threads=True,
            )
            s3_client.upload_file(
                filepath,
                bucket_name,
                key,
                Config=config,
                Callback=ProgressPercentage(filepath=filepath),
            )

        print("")  # print out a newline
        self._update_file_status(file_id=file_id, is_uploading=False, is_complete=True)

    def upload_file_via_api(self, path: Path):
        operations = """{ "query": "mutation uploadFile($input: UploadFileInput!) { uploadFile(input: $input) { ... on File { id } } }", "variables": { "input": { "fileObject": null } } }"""  # noqa
        body = {
            "operations": ("", operations),
            "map": ("", '{"fileObject": ["variables.input.fileObject"]}'),
            "fileObject": (path.name, open(path, "rb")),
        }

        session = create_requests_session(self.spice.host_config)
        transport = self.spice.host_config.get("transport")
        url = f"{transport}://{self.spice.host}/"
        response = session.post(url, files=body)
        return response
