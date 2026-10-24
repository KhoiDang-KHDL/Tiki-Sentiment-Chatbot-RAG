"""Compile both code trees and validate committed configuration without models/keys."""
import json
from pathlib import Path
import tomllib
import yaml


def main():
    root = Path(__file__).resolve().parents[1]
    count = 0
    for directory in ('src', 'runtime', 'scripts', 'tests', 'airflow/dags', 's3_lambda'):
        for path in (root / directory).rglob('*.py'):
            compile(path.read_text(encoding='utf-8-sig'), str(path), 'exec')
            count += 1
    for path in list((root / 'monitoring').rglob('*.yml')) + list((root / '.github/workflows').glob('*.yml')):
        yaml.safe_load(path.read_text(encoding='utf-8'))
    for name in ('docker-compose.yml', 'docker-compose.runtime.yml'):
        yaml.safe_load((root / name).read_text(encoding='utf-8'))
    tomllib.loads((root / '.chainlit/config.toml').read_text(encoding='utf-8'))
    dashboard = json.loads((root / 'monitoring/grafana/dashboards/tiki-api.json').read_text(encoding='utf-8'))
    if dashboard.get('uid') != 'tiki-api' or not dashboard.get('panels'):
        raise ValueError('Invalid dashboard')
    print(f'Compiled {count} Python files; configuration files are valid.')


if __name__ == '__main__':
    main()
