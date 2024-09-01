from Controller.data_processing_control.fetch_files import partition
from dotenv import load_dotenv
import os
load_dotenv()
if __name__ == "__main__":
    print(os.getenv("AWS_S3_URL"))
    print(os.getenv("LOCAL_FILE_DOWNLOAD_DIR"))
    print(os.getenv("AWS_ACCESS_KEY_ID"))
    print(os.getenv("AWS_SECRET_ACCESS_KEY"))
    partition("sample.pdf")
