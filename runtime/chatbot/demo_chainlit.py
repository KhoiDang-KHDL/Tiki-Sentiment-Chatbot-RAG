"""Chainlit entry point. Legacy direct mode: demo_chainlit_local.py."""
from pathlib import Path
import sys
_root = str(Path(__file__).resolve().parents[2])
if _root not in sys.path:
    sys.path.insert(0, _root)
from runtime.chatbot.demo_chainlit_http import *
