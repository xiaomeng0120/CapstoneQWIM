# filepath: c:\Git_Repos\Copilota_Projects\dashboard-QWIM\tests\conftest.py
import pytest
import sys
from pathlib import Path

# Add the src directory to the Python path
@pytest.fixture(scope="session", autouse=True)
def add_src_to_path():
    """Add project root to Python path."""
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))