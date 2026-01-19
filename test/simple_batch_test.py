--"""
Simple test to verify batch query endpoint is working.
"""
import requests
import json

BASE_URL = "http://localhost:8081"

print("Testing batch query endpoint...")
print(f"URL: {BASE_URL}/query/batch\n")

# Simple batch request
batch_request = {
    "queries": [
        "2025 STRATEGY and Q1 REVIEW MEETING ?",
        "Give agenda for the meeting Implementation and Planning Strategy",
        "Compliance Review and Governance Meeting"
    ]
}

try:
    print("Sending batch query request...")
    print(f"Queries: {batch_request['queries']}\n")
    
    response = requests.post(
        f"{BASE_URL}/query/batch",
        json=batch_request,
        timeout=60
    )
    
    print(f"Status Code: {response.status_code}\n")
    
    if response.status_code == 200:
        result = response.json()
        print(f"? SUCCESS!")
        print(f"Total Queries: {result.get('total_queries')}")
        print(f"Successful: {result.get('successful')}")
        print(f"Failed: {result.get('failed')}")
        print(f"Processing Time: {result.get('processing_time', 0):.2f}s\n")
        
        print("Individual Results:")
        print("=" * 60)
        for idx, res in enumerate(result.get('results', [])):
            print(f"\nQuery {idx+1}: {res.get('query')}")
            print(f"Status: {res.get('status')}")
            if res.get('status') == 'success':
                # Get the answer from nested results structure
                query_results = res.get('results', {})
                if query_results:
                    print(f"Answer found in results")
            elif res.get('status') == 'error':
                print(f"Error: {res.get('message')}")
        
    else:
        print(f"? ERROR: {response.status_code}")
        print(response.text)
        
except requests.exceptions.ConnectionError:
    print("? Could not connect to server. Is it running on port 8081?")
except Exception as e:
    print(f"? Error: {e}")
