import os
import urllib.request
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

DATA_DIR = os.path.join(os.path.dirname(__file__), "raw", "cmapss")

# Public HF mirror with the original raw text files
BASE_URL = "https://huggingface.co/datasets/DeveloperMindset123/CMAPSS_Jet_Engine_Simulated_Data/resolve/main"

FILES = [
    "train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt",
    "train_FD002.txt", "test_FD002.txt", "RUL_FD002.txt",
    "train_FD003.txt", "test_FD003.txt", "RUL_FD003.txt",
    "train_FD004.txt", "test_FD004.txt", "RUL_FD004.txt"
]

def download_cmapss():
    os.makedirs(DATA_DIR, exist_ok=True)
    
    for filename in FILES:
        file_path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(file_path):
            print(f"Downloading {filename}...")
            url = f"{BASE_URL}/{filename}"
            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=30) as response, open(file_path, 'wb') as out_file:
                    out_file.write(response.read())
            except Exception as e:
                print(f"Failed to download {filename}: {e}")
                continue
                
    print(f"C-MAPSS dataset ready at {DATA_DIR}")

if __name__ == "__main__":
    download_cmapss()
