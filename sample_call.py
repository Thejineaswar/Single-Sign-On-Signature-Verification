import requests
from sampleBase64 import get_sample_base_64
import time

base64 = get_sample_base_64()
print(base64[:10])
print(len(base64))
# Pass the base64 of 2 images in correspomding keys
request_body = {
    'image_1' : base64,
    'image_2' : base64
}

headers = {"Content-Type": "application/json; charset=utf-8"}


url = "https://41a9-35-233-195-46.ngrok.io/predict"
start_time = time.time()
x = requests.post(url,
                  headers = headers,
                  json = request_body
                  )
print(x.json(),time.time() - start_time)
