import requests


API_KEY = 'your_api_key_here'
url = 'https://api.example.com/data'

headers = {
    'Authorization': f'Bearer {API_KEY}',
    'Content-Type': 'application/json'
}

response = requests.get(url, headers=headers)
print(response.json())