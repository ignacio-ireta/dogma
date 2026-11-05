"""Fixture problems import ``submission`` and are run by the harness, never by the repo's pytest.

Keeping them out of collection stops ``pytest`` from treating their ``test_*.py`` files as tests.
"""

collect_ignore_glob = ["*/test_*.py"]
