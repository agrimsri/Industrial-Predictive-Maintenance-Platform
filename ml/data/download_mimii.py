import os
import urllib.request
import zipfile
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

DATA_DIR = os.path.join(os.path.dirname(__file__), "raw", "mimii", "valve")
# Zenodo links for MIMII valve data. We will download the -6dB SNR file to keep it manageable.
FILES = [
    ("https://zenodo.org/records/3384388/files/-6_dB_valve.zip", "-6_dB_valve.zip"),
]

def download_mimii():
    os.makedirs(DATA_DIR, exist_ok=True)
    
    for url, filename in FILES:
        zip_path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(zip_path):
            print(f"Downloading {filename}...")
            try:
                urllib.request.urlretrieve(url, zip_path)
            except Exception as e:
                print(f"Failed to download {filename}: {e}")
                continue
        
        print(f"Extracting {filename}...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(DATA_DIR)
        except zipfile.BadZipFile:
            print(f"Error: {filename} is not a valid zip file. It may have been corrupted.")
            
    print(f"MIMII dataset ready at {DATA_DIR}")

if __name__ == "__main__":
    download_mimii()
