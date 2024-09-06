import pytest
from .load_test import get_test_data
from fairmofsyncondition.geometry.grapher import SimilarityFinder
from fairmofsyncondition.read_write import coords_library

@pytest.fixture
def test_data():
    """Fixture to load test data."""
    return get_test_data()

@pytest.fixture
def mof5_graph(test_data):
    """Fixture to create graph for MOF5."""
    return coords_library.ase_graph(test_data['MOF5'])

@pytest.fixture
def cof1_graph(test_data):
    """Fixture to create graph for COF1."""
    return coords_library.ase_graph(test_data['COF1'])

@pytest.fixture
def dut8_graph(test_data):
    """Fixture to create graph for DUT8."""
    return coords_library.ase_graph(test_data['DUT8'])

@pytest.mark.parametrize("graph_a, graph_b, expected_similarity", [
    ("mof5_graph", "cof1_graph", 0.078),
    ("mof5_graph", "dut8_graph", 0.146)
])
def test_similarity(graph_a, graph_b, expected_similarity, request):
    """Test the similarity index between different MOF graphs."""
    graph_a = request.getfixturevalue(graph_a)
    graph_b = request.getfixturevalue(graph_b)
    similarity = SimilarityFinder(graph_a, graph_b).compute_similarity_index()
    assert round(similarity, 3) == expected_similarity
