#!/usr/bin/env python3
"""Test with real experiment messages."""

import requests
from src import config

api_key = config.load_api_key()

url = "https://openrouter.ai/api/v1/embeddings"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

# First two seed messages from the experiment
real_messages = [
    'Understand your environment',
    'Look at the process more than the output'
]

data = {
    "model": "openai/text-embedding-3-small",
    "input": real_messages,
    "encoding_format": "float"
}

print("Testing with real experiment messages...")
response = requests.post(url, headers=headers, json=data)
print(f"Status code: {response.status_code}")
result = response.json()
print(f"Response keys: {result.keys()}")
if 'error' in result:
    print(f"Error: {result['error']}")
else:
    print(f"Success! Got {len(result.get('data', []))} embeddings")
