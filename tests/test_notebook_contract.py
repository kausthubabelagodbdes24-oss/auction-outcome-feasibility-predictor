"""Structural checks for the Colab model-selection and export workflow."""

import json
from pathlib import Path


NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "auction_outcome_feasibility_training.ipynb"


def test_notebook_uses_training_only_cv_and_exports_stability_results():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )

    assert "GridSearchCV" in source
    assert "RepeatedStratifiedKFold" in source
    assert '"svm__C": [0.1, 1.0, 10.0, 100.0]' in source
    assert '"svm__gamma": ["scale", 0.01, 0.1, 1.0]' in source
    assert "search.fit(X_train, y_train)" in source
    assert "cross_validate(\n    model,\n    X_train,\n    y_train" in source
    assert '"model_selection": model_selection_report' in source
    assert '"repeated_cross_validation": repeated_cv_report' in source

    selection_cell = "".join(notebook["cells"][17]["source"])
    assert "X_test" not in selection_cell
    assert "y_test" not in selection_cell
