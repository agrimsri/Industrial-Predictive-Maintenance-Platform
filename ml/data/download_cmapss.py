import os
import urllib.request
import zipfile
import ssl

# Disable SSL verification for older servers if necessary
ssl._create_default_https_context = ssl._create_unverified_context

DATA_DIR = os.path.join(os.path.dirname(__file__), "raw", "cmapss")
# We use a known academic/github mirror as the primary NASA link is often under review/unavailable.
# If this fails, users can manually place CMAPSSData.zip into ml/data/raw/cmapss/
URL = "https://raw.githubusercontent.com/luisvalenzuelabay/cmapss/master/CMAPSSData.zip"
FALLBACK_URL = "https://ti.arc.nasa.gov/m/project/prognostic-repository/CMAPSSData.zip"

def download_cmapss():
    os.makedirs(DATA_DIR, exist_ok=True)
    zip_path = os.path.join(DATA_DIR, "CMAPSSData.zip")
    
    if not os.path.exists(zip_path):
        print("Downloading C-MAPSS dataset...")
        try:
            urllib.request.urlretrieve(URL, zip_path)
        except Exception as e:
            print(f"Primary mirror failed: {e}. Trying fallback NASA link...")
            try:
                urllib.request.urlretrieve(FALLBACK_URL, zip_path)
            except Exception as e2:
                print(f"Fallback link failed: {e2}")
                print("Please download CMAPSSData.zip manually from Kaggle or NASA and place it in", DATA_DIR)
                return
    
    print("Extracting C-MAPSS dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR)
    
    print(f"C-MAPSS dataset ready at {DATA_DIR}")

if __name__ == "__main__":
    download_cmapss()
