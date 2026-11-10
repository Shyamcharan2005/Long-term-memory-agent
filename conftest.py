import os
import tempfile

# Must run before any project import, so the agent uses a throwaway data folder
os.environ["MEMORY_DATA_DIR"] = tempfile.mkdtemp(prefix="memtest_")

from contextlib import closing
import pytest
from fact_tools import col
from graph_tools import _conn


@pytest.fixture(autouse=True)
def clean_memory():
    """Wipe both stores before every test."""
    ids = col.get()["ids"]
    if ids:
        col.delete(ids=ids)
    with closing(_conn()) as c:
        c.execute("DELETE FROM relations")
        c.commit()