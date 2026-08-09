import numpy as np
import pytest

from run.pricing_application.run_pricing_application import (
    extract_price,
    response_to_purchase_probability,
)


def test_extract_price_from_pricing_prompt():
    prompt = "The product is priced at: $1,234.50. Would you purchase it?"
    assert extract_price(prompt) == 1234.50


def test_extract_price_requires_exactly_one_price():
    with pytest.raises(ValueError, match="exactly one"):
        extract_price("No price here")
    with pytest.raises(ValueError, match="exactly one"):
        extract_price("priced at: $1.00 and priced at: $2.00")


def test_response_to_purchase_probability_uses_yes_no_orientation_and_clips():
    normalized_response = np.array([-0.2, 0.0, 0.25, 1.0, 1.2])
    expected_purchase = np.array([1.0, 1.0, 0.75, 0.0, 0.0])
    np.testing.assert_allclose(
        response_to_purchase_probability(normalized_response), expected_purchase
    )
