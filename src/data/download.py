import urllib.request, zipfile, io, pathlib

RAW = pathlib.Path("data/raw")
RAW.mkdir(parents=True, exist_ok=True)

URLS = {
    "household": "https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip",
    "appliances": "https://archive.ics.uci.edu/static/public/374/appliances+energy+prediction.zip",
}

def download_all():
    for name, url in URLS.items():
        print(f"Téléchargement {name}...")
        data = urllib.request.urlopen(url).read()
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            z.extractall(RAW / name)

if __name__ == "__main__":
    download_all()