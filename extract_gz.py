import tarfile

def extract_gz(tar_path, extract_path='.'):
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=extract_path)