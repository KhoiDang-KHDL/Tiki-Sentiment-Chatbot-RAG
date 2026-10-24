"""Read-only checks from the UI network to Prometheus and Grafana."""
import json
import urllib.request
from urllib.parse import urlencode


def read(url):
    return json.load(urllib.request.urlopen(url, timeout=15))


def main():
    health = read('http://grafana:3000/api/health')
    targets = read('http://prometheus:9090/api/v1/targets')['data']['activeTargets']
    real = [t for t in targets if t['labels']['job'] == 'tiki-api']
    if not real or any(t['health'] != 'up' for t in real):
        raise RuntimeError('Prometheus target tiki-api is not UP')
    metrics = read('http://prometheus:9090/api/v1/query?' + urlencode({'query':'up{job="tiki-api"}'}))
    print(json.dumps({'grafana_database':health['database'], 'api_scrape':'up',
                      'query_result':metrics['data']['result']}, indent=2))


if __name__ == '__main__':
    main()
