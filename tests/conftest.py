import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

#load_candidates выполянем один раз ща весь запуск pytest
#он нигде не меняется, поэтому никакие мутация не стращны
@pytest.fixture(scope="session")
def candidates():
    from src.candidates import load_candidates
    return load_candidates()
