"""Run the offline suite with a network guard and portable result artifacts."""
import json
import ipaddress
import os
from pathlib import Path
import socket
import sys
import unittest
from unittest.mock import patch


def deny_connection(sock, address):
    # Windows implements asyncio's socketpair with loopback TCP sockets.
    if os.name == 'nt' and sock.family in (socket.AF_INET, socket.AF_INET6):
        try:
            if ipaddress.ip_address(address[0]).is_loopback:
                return ORIGINAL_CONNECT(sock, address)
        except ValueError:
            pass
    if sock.family in (socket.AF_INET, socket.AF_INET6):
        raise RuntimeError('Offline CI forbids network connections; mock this request')
    return ORIGINAL_CONNECT(sock, address)


ORIGINAL_CONNECT = socket.socket.connect


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / 'reports' / 'ci'
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(root / 'tests'))
    with (output / 'unit-tests.txt').open('w', encoding='utf-8') as log:
        with patch.object(socket.socket, 'connect', deny_connection):
            suite = unittest.defaultTestLoader.discover(str(root / 'tests'))
            result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    report = {'tests': result.testsRun, 'failures': len(result.failures),
              'errors': len(result.errors), 'skipped': len(result.skipped),
              'passed': result.wasSuccessful() and result.testsRun > 0,
              'skip_reasons': [reason for _, reason in result.skipped]}
    (output / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print((output / 'unit-tests.txt').read_text(encoding='utf-8'))
    print(json.dumps(report, indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
