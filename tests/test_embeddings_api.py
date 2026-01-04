#!/usr/bin/env python3
"""Test OpenRouter embeddings API."""

import requests
from src import config

api_key = config.load_api_key()

# Test 1: Try the model from the example
url = "https://openrouter.ai/api/v1/embeddings"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

data_single = {
    "model": "openai/text-embedding-3-small",
    "input": "test message",
    "encoding_format": "float"
}

data_batch = {
    "model": "openai/text-embedding-3-small",
    "input": ["test message 1", "test message 2", "test message 3"],
    "encoding_format": "float"
}

print("Testing OpenRouter embeddings API...")
print(f"URL: {url}")
print()

print("Test 1: Single message")
response = requests.post(url, headers=headers, json=data_single)
print(f"Status code: {response.status_code}")
result = response.json()
print(f"Response keys: {result.keys()}")
print(f"Num embeddings: {len(result.get('data', []))}")
print()

print("Test 2: Batch (3 messages)")
response = requests.post(url, headers=headers, json=data_batch)
print(f"Status code: {response.status_code}")
result = response.json()
print(f"Response keys: {result.keys()}")
print(f"Num embeddings: {len(result.get('data', []))}")
print(f"Response: {result}")
