import requests
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv('API_KEY')
url = 'https://api.collegebasketballdata.com/'

headers = {
    'Authorization': f'Bearer {API_KEY}',
    'Content-Type': 'application/json'
}

response = requests.get(url, headers=headers)

# Debug: Check the status code and response text
print(f"Status Code: {response.status_code}")
print(f"Response Text: {response.text}")

# Only try to parse JSON if we got a successful response
if response.status_code == 200:
    try:
        print(response.json())
    except:
        print("Response is not valid JSON")