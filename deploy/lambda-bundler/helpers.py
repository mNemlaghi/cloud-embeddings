from pathlib import Path
import os
import sys
import shutil
from huggingface_hub import snapshot_download
import tarfile
import boto3

HF_MODEL_ID="sentence-transformers/all-mpnet-base-v2"
# set HF_HUB_ENABLE_HF_TRANSFER env var to enable hf-transfer for faster downloads
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

class HfModelBuilder():
    
    def __init__(self, hf_model_id, model_name):
        self.current_code_path =  Path(sys.path[0] + "/inference.py")
        os.chdir("/tmp")
        self.hf_model_id=hf_model_id
        self.model_name = model_name
        self.model_directory = Path("/tmp/"+self.model_name)
        shutil.rmtree(self.model_directory, ignore_errors = True)
        self.model_directory.mkdir(exist_ok=True)
        
    def download_model_from_hf(self):
        # Download model
        os.chdir("/tmp")
        snapshot_download(self.hf_model_id,
            local_dir=str(self.model_directory), # download to model dir
            revision="main", # use a specific revision, e.g. refs/pr/21
            ignore_patterns=["onnx*", "*safetensors"],
            local_dir_use_symlinks=False # use no symlinks to save disk space
        )

    def copy_custom_inference_code_to_model_directory(self):
        dest = self.model_directory / "code"
        dest.mkdir(exist_ok=True)
        shutil.copy(self.current_code_path, dest)

    def archive_directory(self):
        self.owd = os.getcwd()
        os.chdir(self.model_directory)
        self.model_archive_path=Path("model.tar.gz")
        p = self.model_directory.rglob('*')
        with tarfile.open(self.model_archive_path.resolve(), "w:gz", compresslevel=5) as tar:
            for root, dirs, files in os.walk(".", topdown=False):
                for name in files:
                    tar.add(os.path.join(root, name))

    def move_up_archive_and_remove_original_dir(self):
        p = self.model_archive_path.absolute()
        parent_dir = p.parents[1]
        p.rename(parent_dir / p.name)
        os.chdir(self.owd)
        shutil.rmtree(self.model_directory.resolve(), ignore_errors = False) #Needs to be existing first

    def send_to_s3(self, bucket_name):
        s3 =boto3.client('s3')
        s3_dest_uri = "s3://"+bucket_name + "/models/"+self.model_name+"/model.tar.gz"
        s3.upload_file(self.model_archive_path.resolve(),bucket_name, "models/"+self.model_name+"/model.tar.gz")
        return s3_dest_uri

    @classmethod
    def run_from_hf_model_id(cls, hf_model_id, bucket_name, model_name):
        builder = cls(hf_model_id, model_name)
        builder.download_model_from_hf()
        builder.copy_custom_inference_code_to_model_directory()
        builder.archive_directory()
        builder.move_up_archive_and_remove_original_dir()
        s3_uri =builder.send_to_s3(bucket_name)
        return s3_uri


if __name__=='__main__':
    builder = HfModelBuilder(HF_MODEL_ID)
    builder.download_model_from_hf()
    builder.copy_custom_inference_code_to_model_directory()
    builder.archive_directory()
    builder.move_up_archive_and_remove_original_dir()
    print(builder.model_archive_path.resolve())
