import os
import urllib.request
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

DATA_DIR = os.path.join(os.path.dirname(__file__), "raw", "cwru")
# URLs from Case Western Reserve University Bearing Data Center
# 97.mat: Normal Baseline Data (0 HP, 1797 RPM)
# 105.mat: 12k Drive End Bearing Fault Data (0.007" Inner Race, 0 HP, 1797 RPM)
FILES = [
    ("https://engineering.case.edu/sites/default/files/97.mat", "97.mat"),
    ("https://engineering.case.edu/sites/default/files/105.mat", "105.mat")
]

def download_cwru():
    os.makedirs(DATA_DIR, exist_ok=True)
    
    for url, filename in FILES:
        file_path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(file_path):
            print(f"Downloading {filename}...")
            try:
                # Add headers to avoid 403 Forbidden
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req) as response, open(file_path, 'wb') as out_file:
                    out_file.write(response.read())
            except Exception as e:
                print(f"Failed to download {filename}: {e}")
                continue
            
    print(f"CWRU dataset ready at {DATA_DIR}")

if __name__ == "__main__":
    download_cwru()
