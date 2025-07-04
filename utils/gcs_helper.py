from google.cloud import storage
from streamlit.runtime.uploaded_file_manager import UploadedFile, UploadedFileRec

def download_from_gcs(bucket_name: str, blob_path: str) -> UploadedFile:
    """
    Download raw bytes from GCS into memory.
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob   = bucket.blob(blob_path)

    if not blob.exists(client): return None

    bytes = blob.download_as_bytes()
    file_name = blob_path.split("/")[-1]
    file_type = blob_path.split(".")[-1]

    uploaded_file_rec = UploadedFileRec(file_id=f"gs://{bucket_name}/{blob_path}", name=file_name, type=file_type, data=bytes)

    return UploadedFile(record=uploaded_file_rec, file_urls=None)