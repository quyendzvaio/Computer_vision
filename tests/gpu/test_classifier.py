"""Test PPEClassifier configuration and preprocess logic."""
import numpy as np
import pytest
from gpu.classifier import PPEClassifier, CLASS_NAMES


def test_unknown_item_raises():
    with pytest.raises(ValueError):
        PPEClassifier("hat")


def test_class_names_exist():
    assert "helmet" in CLASS_NAMES
    assert "vest" in CLASS_NAMES
    assert "boot" in CLASS_NAMES
    for names in CLASS_NAMES.values():
        assert len(names) == 2
