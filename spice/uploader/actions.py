import hashlib
from pathlib import Path
import sys
import threading
from typing import Dict, Optional

import boto3
from boto3.s3.transfer import TransferConfig
import botocore.exceptions
from gql import gql


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

    def _create_file(self, file_name: str, file_size: int, file_checksum: str):
        mutation = gql(
            """
            mutation createFile(
            $fileName: String!
            $fileSize: Int!
            $fileChecksum: String!
            ) {
                createFile(fileName: $fileName, fileSize: $fileSize, fileChecksum: $fileChecksum) {
                    id
                }
            }
        """  # noqa
        )
        variables = {
            "fileName": file_name,
            "fileSize": file_size,
            "fileChecksum": file_checksum,
        }
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
            mutation updateFileStatus($fileId: String!, $isUploading: Boolean, $isComplete: Boolean) {
                updateFileStatus(fileId: $fileId, isUploading: $isUploading, isComplete: $isComplete) {
                    id
                }
            }
        """  # noqa
        )
        variables: Dict[str, str | bool] = {
            "fileId": file_id,
        }
        if is_uploading is not None:
            variables["isUploading"] = is_uploading
        if is_complete is not None:
            variables["isComplete"] = is_complete

        result = self.spice.session.execute(mutation, variable_values=variables)
        return result

    def upload_file(
        self,
        bucket_name: str,
        key: str,
        filepath: Path,
        file_id: Optional[str],
        overwrite: bool = False,
    ):
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
            )

        s3_resource = boto3.resource("s3")
        s3_client = boto3.client("s3")
        # TODO: configure these values based on the available system properties
        # if a file is bigger than multipart_threshold, then do multipart upload
        multipart_threshold = 1024 * 100
        multipart_chunksize = 1024 * 100
        max_concurrency = 16
        config = TransferConfig(
            multipart_threshold=multipart_threshold,
            max_concurrency=max_concurrency,
            multipart_chunksize=multipart_chunksize,  # 25MB
            use_threads=True,
        )

        self._update_file_status(file_id=file_id, is_uploading=True, is_complete=False)

        file_already_uploaded = False
        try:
            # TODO: add a check here that the checksum is the same
            head_object = s3_client.head_object(Bucket=bucket_name, Key=key)
            file_already_uploaded = True
            print("File already uploaded.")
        except botocore.exceptions.ClientError as exception:
            if exception.response["Error"]["Code"] == "404":
                # The key does not exist and we should upload the file
                pass
            elif exception.response["Error"]["Code"] == 403:
                # Unauthorized, including invalid bucket
                raise exception
            else:
                raise exception

        if (file_already_uploaded is False) or (
            file_already_uploaded and overwrite is True
        ):
            s3_resource.Object(bucket_name, key).upload_file(
                filepath, Config=config, Callback=ProgressPercentage(filepath=filepath)
            )
        print("")
        self._update_file_status(file_id=file_id, is_uploading=False, is_complete=True)
