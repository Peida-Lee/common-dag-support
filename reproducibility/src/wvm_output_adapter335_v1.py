"""Preserve author WVM decisions for a scalar or a vector of levels."""

import json
from pathlib import Path
import tomllib

import numpy as np


def level_rows(flags, levels, predictors):
    if len(levels) == 1 and len(flags) == predictors and all(isinstance(value, bool) for value in flags):
        flags = [flags]
    if len(flags) != len(levels) or any(len(row) != predictors for row in flags):
        raise ValueError("Author cause dimensions disagree with levels and predictors")
    if not all(isinstance(value, bool) for row in flags for value in row):
        raise ValueError("Author causes must be Boolean")
    return flags


def fit_wvm_output(task, inherited_fit):
    native = Path(task["output"]).parent / "author_result.toml"
    if not native.exists():
        try:
            inherited_fit(task)
        except TypeError:
            if not native.exists():
                raise
    author = tomllib.loads(native.read_text())
    config = tomllib.loads(Path(task["config"]).read_text())
    if author["status"] != "completed" or author["source_modified"]:
        raise ValueError("A completed unmodified-author output is required")
    predictors, response = config["predictor_columns"], config["response_column"]
    flags = level_rows(author["author_causes_by_level"], config["alphas"], len(predictors))
    assert author["group_counts"] == config["counts"] and author["rows"] == sum(config["counts"])
    assert author["predictor_columns"] == predictors and author["response_column"] == response
    assert author["B"] == config["B"] and author["seed"] == config["seed"]
    for key in ("raw_pvalues", "author_thresholds", "exclusion_scores", "author_global_minimum"):
        assert np.isfinite(np.asarray(author[key], dtype=float)).all()
    assert len(author["raw_pvalues"]) == len(predictors)
    assert len(author["author_thresholds"]) == len(config["alphas"])
    native_graphs = []
    for index, selected in enumerate(flags):
        score_flags = [value > author["author_thresholds"][index] for value in author["exclusion_scores"]]
        assert selected == score_flags
        native_graphs.append([[parent - 1, response - 1] for parent, active in zip(predictors, selected) if active])
    return dict(status="completed", method="WVM_author", native_edges=native_graphs[0],
                native_undirected_edges=[], author=author, truth_read=False,
                metadata=dict(source_modified=False, B=author["B"], levels=config["alphas"],
                              native_graphs_by_level=native_graphs,
                              author_causes_shape="single_level_vector" if len(config["alphas"]) == 1 else "multiple_level_rows",
                              output_adapter="wvm_output_adapter335_v1.py", optimization_replayed=False))


