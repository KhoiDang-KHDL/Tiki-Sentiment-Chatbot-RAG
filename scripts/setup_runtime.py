"""Create private optional-runtime configuration without overwriting existing secrets."""
from pathlib import Path
import secrets


def main():
    root = Path(__file__).resolve().parents[1]
    target = root / '.env.runtime'
    if not target.exists():
        template = (root / '.env.runtime.example').read_text(encoding='utf-8')
        target.write_text(template.replace('CHANGE_ME_BEFORE_START', secrets.token_hex(32)), encoding='utf-8')
        print('Created .env.runtime with a random Grafana password (not printed).')
    else:
        print('Keeping existing .env.runtime.')
    for directory in ('reports/loadtest', 'logs', 'models', 'data'):
        (root / directory).mkdir(parents=True, exist_ok=True)


if __name__ == '__main__':
    main()
