"""
Entry point for python -m dataset_forensics.

Usage:
    python -m dataset_forensics --config config.yaml
    python -m dataset_forensics --help
"""

from dataset_forensics.cli import main

if __name__ == '__main__':
    import sys
    sys.exit(main())
