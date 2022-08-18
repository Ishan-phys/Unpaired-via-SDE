import requests, zipfile
from io import BytesIO
url = "http://efrosgans.eecs.berkeley.edu/cyclegan/datasets/horse2zebra.zip"
filename = url.split('/')[-1]

req = requests.get(url)

# extracting the zip file contents
zipfile = zipfile.ZipFile(BytesIO(req.content))
zipfile.extractall('./datasets')