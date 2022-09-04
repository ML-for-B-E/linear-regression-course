import numpy as np
import pandas as pd
from reglin.metrics.errors import calculer_erreur

def test_calculer_erreur_returns_correct_values():
    # Given
    

    df = pd.DataFrame({"Y": [2,10,5],"X": [0.5,-62,2]})
    beta = [2,6]
    predictor_col = "X"
    target_col = "Y"
    expected_quadratic_error = 74.0135123
    # When
    computed_quadratic_error = calculer_erreur(beta, df, predictor_col, target_col)

    # Given
    np.testing.assert_allclose(computed_quadratic_error, expected_quadratic_error)

