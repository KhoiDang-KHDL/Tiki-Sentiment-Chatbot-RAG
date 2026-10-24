"""Read-only smoke check from UI to API to Qdrant. Does not spend LLM quota."""
import json
import os
import urllib.request


def main():
    base = os.environ.get('API_BASE_URL', 'http://api:8000').rstrip('/')
    health = json.load(urllib.request.urlopen(base + '/health', timeout=20))
    payload = json.dumps({'query': 'sản phẩm chất lượng tốt', 'top_k': 2}).encode('utf-8')
    request = urllib.request.Request(base + '/search', data=payload,
                                     headers={'Content-Type': 'application/json'})
    result = json.load(urllib.request.urlopen(request, timeout=180))
    assert result['documents'], 'Search returned no documents'
    print(json.dumps({'api': base, 'health': health, 'search_documents': len(result['documents']),
                      'model_in_ui': False}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
