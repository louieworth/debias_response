"""Reproducible hyperparameter search for OpinionQA individual models.

The default mode uses an outer split carved from the published training set.
Passing ``--eval-split test`` instead fits on the full published training set
and ranks configurations directly on the published test set.  The latter is
an explicitly test-directed exploratory analysis and must be reported as such.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from debias.debias_variants import VariantDebiasModel, prepare_features
from debias.evaluate_variants import (
    compute_accuracy_hard,
    compute_accuracy_mad,
    compute_accuracy_soft,
)


TRAIN_PATH = PROJECT_ROOT / "dataset/OpinionQA/individual/individual_train.parquet"
TEST_PATH = PROJECT_ROOT / "dataset/OpinionQA/individual/individual_test.parquet"

CURRENT_LAYERS = "6144,3072,1536,768,384"
MEDIUM_LAYERS = "4096,2048,1024,512,256"
SMALL_LAYERS = "3072,1536,768,384"
COMPACT_LAYERS = "2048,1024,512,256"


def config(
    name,
    layers=CURRENT_LAYERS,
    alpha=1e-4,
    learning_rate=2e-4,
    dropout=0.05,
    batch_size=512,
    validation_fraction=0.1,
    n_iter_no_change=60,
    standardize=True,
    vector_transform="raw",
):
    return {
        "name": name,
        "hidden_layers": layers,
        "alpha": alpha,
        "learning_rate": learning_rate,
        "dropout": dropout,
        "batch_size": batch_size,
        "validation_fraction": validation_fraction,
        "n_iter_no_change": n_iter_no_change,
        "standardize": standardize,
        "vector_transform": vector_transform,
    }


# A bounded neighborhood around the published individual-level configuration.
# Config 0 exactly reproduces the current hyperparameters.
CONFIGS = [
    config("current"),
    config("current_a1e-3", alpha=1e-3),
    config("current_a1e-2", alpha=1e-2),
    config("current_d10", dropout=0.10),
    config("current_d20", dropout=0.20),
    config("current_lr1e-4", learning_rate=1e-4),
    config("current_lr5e-4", learning_rate=5e-4),
    config("current_b256", batch_size=256),
    config("medium", layers=MEDIUM_LAYERS),
    config("medium_a1e-3_d10", layers=MEDIUM_LAYERS, alpha=1e-3, dropout=0.10),
    config("medium_a1e-2_d10", layers=MEDIUM_LAYERS, alpha=1e-2, dropout=0.10),
    config(
        "medium_a1e-3_lr1e-4_d10",
        layers=MEDIUM_LAYERS,
        alpha=1e-3,
        learning_rate=1e-4,
        dropout=0.10,
    ),
    config("medium_a5e-4", layers=MEDIUM_LAYERS, alpha=5e-4),
    config("medium_a1e-3_d20", layers=MEDIUM_LAYERS, alpha=1e-3, dropout=0.20),
    config("small", layers=SMALL_LAYERS),
    config("small_a1e-3_d10", layers=SMALL_LAYERS, alpha=1e-3, dropout=0.10),
    config("small_a1e-2_d10", layers=SMALL_LAYERS, alpha=1e-2, dropout=0.10),
    config(
        "small_a1e-3_lr1e-4_d10",
        layers=SMALL_LAYERS,
        alpha=1e-3,
        learning_rate=1e-4,
        dropout=0.10,
    ),
    config("small_a5e-4", layers=SMALL_LAYERS, alpha=5e-4),
    config("compact", layers=COMPACT_LAYERS),
    config("compact_a1e-3_d10", layers=COMPACT_LAYERS, alpha=1e-3, dropout=0.10),
    config("compact_a1e-2_d10", layers=COMPACT_LAYERS, alpha=1e-2, dropout=0.10),
    config(
        "compact_a1e-3_lr1e-4_d10",
        layers=COMPACT_LAYERS,
        alpha=1e-3,
        learning_rate=1e-4,
        dropout=0.10,
    ),
    config("compact_a5e-4", layers=COMPACT_LAYERS, alpha=5e-4),
    config(
        "medium_a1e-3_d10_b256",
        layers=MEDIUM_LAYERS,
        alpha=1e-3,
        dropout=0.10,
        batch_size=256,
    ),
    config(
        "small_a1e-3_d10_b256",
        layers=SMALL_LAYERS,
        alpha=1e-3,
        dropout=0.10,
        batch_size=256,
    ),
    config(
        "compact_a1e-3_d10_b256",
        layers=COMPACT_LAYERS,
        alpha=1e-3,
        dropout=0.10,
        batch_size=256,
    ),
    config(
        "medium_a1e-3_lr5e-4_d10",
        layers=MEDIUM_LAYERS,
        alpha=1e-3,
        learning_rate=5e-4,
        dropout=0.10,
    ),
    config(
        "small_a1e-3_lr5e-4_d10",
        layers=SMALL_LAYERS,
        alpha=1e-3,
        learning_rate=5e-4,
        dropout=0.10,
    ),
    config("current_a5e-4", alpha=5e-4),
    config(
        "small_a1e-2_d10_val05",
        layers=SMALL_LAYERS,
        alpha=1e-2,
        dropout=0.10,
        validation_fraction=0.05,
    ),
    config(
        "small_a1e-2_d10_val15",
        layers=SMALL_LAYERS,
        alpha=1e-2,
        dropout=0.10,
        validation_fraction=0.15,
    ),
    config(
        "small_a1e-2_d10_val20",
        layers=SMALL_LAYERS,
        alpha=1e-2,
        dropout=0.10,
        validation_fraction=0.20,
    ),
    config(
        "small_a1e-2_d10_pat30",
        layers=SMALL_LAYERS,
        alpha=1e-2,
        dropout=0.10,
        n_iter_no_change=30,
    ),
    config(
        "small_a1e-2_d10_pat100",
        layers=SMALL_LAYERS,
        alpha=1e-2,
        dropout=0.10,
        n_iter_no_change=100,
    ),
    config(
        "small_a1e-2_d10_b256",
        layers=SMALL_LAYERS,
        alpha=1e-2,
        dropout=0.10,
        batch_size=256,
    ),
    config(
        "small_a1e-2_d10_b1024",
        layers=SMALL_LAYERS,
        alpha=1e-2,
        dropout=0.10,
        batch_size=1024,
    ),
    config(
        "small_a1e-2_lr1e-4_d10",
        layers=SMALL_LAYERS,
        alpha=1e-2,
        learning_rate=1e-4,
        dropout=0.10,
    ),
    config(
        "small_a1e-2_lr3e-4_d10",
        layers=SMALL_LAYERS,
        alpha=1e-2,
        learning_rate=3e-4,
        dropout=0.10,
    ),
    config("small_a1e-2_d05", layers=SMALL_LAYERS, alpha=1e-2, dropout=0.05),
    config("small_a1e-2_d15", layers=SMALL_LAYERS, alpha=1e-2, dropout=0.15),
    config("small_a1e-2_d20", layers=SMALL_LAYERS, alpha=1e-2, dropout=0.20),
    config(
        "small_a1e-2_d10_no_std",
        layers=SMALL_LAYERS,
        alpha=1e-2,
        dropout=0.10,
        standardize=False,
    ),
    config(
        "medium_a1e-2_d10_val05",
        layers=MEDIUM_LAYERS,
        alpha=1e-2,
        dropout=0.10,
        validation_fraction=0.05,
    ),
    config(
        "compact_a1e-2_d10_val05",
        layers=COMPACT_LAYERS,
        alpha=1e-2,
        dropout=0.10,
        validation_fraction=0.05,
    ),
    config("small_a3e-3_d10", layers=SMALL_LAYERS, alpha=3e-3, dropout=0.10),
    config("small_a5e-3_d10", layers=SMALL_LAYERS, alpha=5e-3, dropout=0.10),
    config("small_a2e-2_d10", layers=SMALL_LAYERS, alpha=2e-2, dropout=0.10),
    config("small_a3e-2_d10", layers=SMALL_LAYERS, alpha=3e-2, dropout=0.10),
    config(
        "width3584_a1e-2_d10",
        layers="3584,1792,896,448",
        alpha=1e-2,
        dropout=0.10,
    ),
    config(
        "width2560_a1e-2_d10",
        layers="2560,1280,640,320",
        alpha=1e-2,
        dropout=0.10,
    ),
    config(
        "width1536_a1e-2_d10",
        layers="1536,768,384,192",
        alpha=1e-2,
        dropout=0.10,
    ),
    config(
        "medium4_a1e-2_d10",
        layers="4096,2048,1024,512",
        alpha=1e-2,
        dropout=0.10,
    ),
    config(
        "small3_a1e-2_d10",
        layers="3072,1536,768",
        alpha=1e-2,
        dropout=0.10,
    ),
    config(
        "compact3_a1e-2_d10",
        layers="2048,1024,512",
        alpha=1e-2,
        dropout=0.10,
    ),
    config(
        "small_a1e-2_d10_sorted",
        layers=SMALL_LAYERS,
        alpha=1e-2,
        dropout=0.10,
        vector_transform="sorted",
    ),
    config(
        "small_a1e-2_d10_centered_sorted",
        layers=SMALL_LAYERS,
        alpha=1e-2,
        dropout=0.10,
        vector_transform="centered_sorted",
    ),
    config(
        "small_a1e-2_d10_mean_centered_sorted",
        layers=SMALL_LAYERS,
        alpha=1e-2,
        dropout=0.10,
        vector_transform="mean_centered_sorted",
    ),
    config(
        "medium_a1e-2_d10_sorted",
        layers=MEDIUM_LAYERS,
        alpha=1e-2,
        dropout=0.10,
        vector_transform="sorted",
    ),
    config(
        "medium_a1e-2_d10_centered_sorted",
        layers=MEDIUM_LAYERS,
        alpha=1e-2,
        dropout=0.10,
        vector_transform="centered_sorted",
    ),
    config(
        "medium_a1e-2_d10_mean_centered_sorted",
        layers=MEDIUM_LAYERS,
        alpha=1e-2,
        dropout=0.10,
        vector_transform="mean_centered_sorted",
    ),
    config(
        "compact_a1e-2_d10_sorted",
        layers=COMPACT_LAYERS,
        alpha=1e-2,
        dropout=0.10,
        vector_transform="sorted",
    ),
    config(
        "compact_a1e-2_d10_centered_sorted",
        layers=COMPACT_LAYERS,
        alpha=1e-2,
        dropout=0.10,
        vector_transform="centered_sorted",
    ),
    config(
        "compact_a1e-2_d10_mean_centered_sorted",
        layers=COMPACT_LAYERS,
        alpha=1e-2,
        dropout=0.10,
        vector_transform="mean_centered_sorted",
    ),
    config(
        "small_a3e-2_d10_sorted",
        layers=SMALL_LAYERS,
        alpha=3e-2,
        dropout=0.10,
        vector_transform="sorted",
    ),
    config(
        "small_a3e-2_d10_mean_centered_sorted",
        layers=SMALL_LAYERS,
        alpha=3e-2,
        dropout=0.10,
        vector_transform="mean_centered_sorted",
    ),
    config(
        "small_a5e-3_d10_mean_centered_sorted",
        layers=SMALL_LAYERS,
        alpha=5e-3,
        dropout=0.10,
        vector_transform="mean_centered_sorted",
    ),
    config(
        "small_a2e-2_d10_mean_centered_sorted",
        layers=SMALL_LAYERS,
        alpha=2e-2,
        dropout=0.10,
        vector_transform="mean_centered_sorted",
    ),
    config(
        "small_a1e-2_d20_mean_centered_sorted",
        layers=SMALL_LAYERS,
        alpha=1e-2,
        dropout=0.20,
        vector_transform="mean_centered_sorted",
    ),
    config(
        "small_a1e-2_d00_mean_centered_sorted",
        layers=SMALL_LAYERS,
        alpha=1e-2,
        dropout=0.0,
        vector_transform="mean_centered_sorted",
    ),
    config("two2048_a1e-2_d10", layers="2048,1024", alpha=1e-2, dropout=0.10),
    config("two1024_a1e-2_d10", layers="1024,512", alpha=1e-2, dropout=0.10),
    config("two512_a1e-2_d10", layers="512,256", alpha=1e-2, dropout=0.10),
    config("three1024_a1e-2_d10", layers="1024,512,256", alpha=1e-2, dropout=0.10),
    config("two2048_a3e-2_d10", layers="2048,1024", alpha=3e-2, dropout=0.10),
    config("two1024_a3e-2_d10", layers="1024,512", alpha=3e-2, dropout=0.10),
    config("two512_a3e-2_d10", layers="512,256", alpha=3e-2, dropout=0.10),
    config("one1024_a1e-2_d10", layers="1024", alpha=1e-2, dropout=0.10),
    config("small_a1e-1_d10", layers=SMALL_LAYERS, alpha=1e-1, dropout=0.10),
    config("compact_a1e-1_d10", layers=COMPACT_LAYERS, alpha=1e-1, dropout=0.10),
    # Test-directed refinement around config 35, whose five-seed Vector/One
    # Logprob gap is small.  These keep the same architecture and vary one
    # optimization/regularization axis at a time.
    config(
        "small_a3e-3_d10_b256",
        layers=SMALL_LAYERS,
        alpha=3e-3,
        dropout=0.10,
        batch_size=256,
    ),
    config(
        "small_a5e-3_d10_b256",
        layers=SMALL_LAYERS,
        alpha=5e-3,
        dropout=0.10,
        batch_size=256,
    ),
    config(
        "small_a7p5e-3_d10_b256",
        layers=SMALL_LAYERS,
        alpha=7.5e-3,
        dropout=0.10,
        batch_size=256,
    ),
    config(
        "small_a1p5e-2_d10_b256",
        layers=SMALL_LAYERS,
        alpha=1.5e-2,
        dropout=0.10,
        batch_size=256,
    ),
    config(
        "small_a2e-2_d10_b256",
        layers=SMALL_LAYERS,
        alpha=2e-2,
        dropout=0.10,
        batch_size=256,
    ),
    config(
        "small_a1e-2_d05_b256",
        layers=SMALL_LAYERS,
        alpha=1e-2,
        dropout=0.05,
        batch_size=256,
    ),
    config(
        "small_a1e-2_d08_b256",
        layers=SMALL_LAYERS,
        alpha=1e-2,
        dropout=0.08,
        batch_size=256,
    ),
    config(
        "small_a1e-2_d12_b256",
        layers=SMALL_LAYERS,
        alpha=1e-2,
        dropout=0.12,
        batch_size=256,
    ),
    config(
        "small_a1e-2_d15_b256",
        layers=SMALL_LAYERS,
        alpha=1e-2,
        dropout=0.15,
        batch_size=256,
    ),
    config(
        "small_a1e-2_d20_b256",
        layers=SMALL_LAYERS,
        alpha=1e-2,
        dropout=0.20,
        batch_size=256,
    ),
    config(
        "small_a1e-2_lr1e-4_d10_b256",
        layers=SMALL_LAYERS,
        alpha=1e-2,
        learning_rate=1e-4,
        dropout=0.10,
        batch_size=256,
    ),
    config(
        "small_a1e-2_lr3e-4_d10_b256",
        layers=SMALL_LAYERS,
        alpha=1e-2,
        learning_rate=3e-4,
        dropout=0.10,
        batch_size=256,
    ),
    # Refinement around config 69, for which a mean-centered sorted Vector
    # produced the strongest single-seed margin.
    config(
        "small_a3e-3_d00_b512_mean_centered_sorted",
        layers=SMALL_LAYERS,
        alpha=3e-3,
        dropout=0.0,
        vector_transform="mean_centered_sorted",
    ),
    config(
        "small_a5e-3_d00_b512_mean_centered_sorted",
        layers=SMALL_LAYERS,
        alpha=5e-3,
        dropout=0.0,
        vector_transform="mean_centered_sorted",
    ),
    config(
        "small_a7p5e-3_d00_b512_mean_centered_sorted",
        layers=SMALL_LAYERS,
        alpha=7.5e-3,
        dropout=0.0,
        vector_transform="mean_centered_sorted",
    ),
    config(
        "small_a1p5e-2_d00_b512_mean_centered_sorted",
        layers=SMALL_LAYERS,
        alpha=1.5e-2,
        dropout=0.0,
        vector_transform="mean_centered_sorted",
    ),
    config(
        "small_a2e-2_d00_b512_mean_centered_sorted",
        layers=SMALL_LAYERS,
        alpha=2e-2,
        dropout=0.0,
        vector_transform="mean_centered_sorted",
    ),
    config(
        "small_a1e-2_d02_b512_mean_centered_sorted",
        layers=SMALL_LAYERS,
        alpha=1e-2,
        dropout=0.02,
        vector_transform="mean_centered_sorted",
    ),
    config(
        "small_a1e-2_d05_b512_mean_centered_sorted",
        layers=SMALL_LAYERS,
        alpha=1e-2,
        dropout=0.05,
        vector_transform="mean_centered_sorted",
    ),
    config(
        "small_a1e-2_d00_b256_mean_centered_sorted",
        layers=SMALL_LAYERS,
        alpha=1e-2,
        dropout=0.0,
        batch_size=256,
        vector_transform="mean_centered_sorted",
    ),
    # Full-training configurations: validation_fraction=0 avoids holding out a
    # temporary subset of the official training split.  Early stopping then
    # monitors training loss, while model selection is based on the test metric
    # as explicitly requested for this exploratory search.
    config(
        "small_a1e-2_d10_b512_fulltrain",
        layers=SMALL_LAYERS,
        alpha=1e-2,
        dropout=0.10,
        validation_fraction=0.0,
        n_iter_no_change=30,
    ),
    config(
        "small_a1e-2_d10_b256_fulltrain",
        layers=SMALL_LAYERS,
        alpha=1e-2,
        dropout=0.10,
        batch_size=256,
        validation_fraction=0.0,
        n_iter_no_change=30,
    ),
    config(
        "small_a3e-3_d10_b256_fulltrain",
        layers=SMALL_LAYERS,
        alpha=3e-3,
        dropout=0.10,
        batch_size=256,
        validation_fraction=0.0,
        n_iter_no_change=30,
    ),
    config(
        "small_a1e-2_d12_b256_fulltrain",
        layers=SMALL_LAYERS,
        alpha=1e-2,
        dropout=0.12,
        batch_size=256,
        validation_fraction=0.0,
        n_iter_no_change=30,
    ),
    config(
        "small_a1e-2_d15_b512_fulltrain",
        layers=SMALL_LAYERS,
        alpha=1e-2,
        dropout=0.15,
        validation_fraction=0.0,
        n_iter_no_change=30,
    ),
    config(
        "small_a3e-3_d10_b512_fulltrain",
        layers=SMALL_LAYERS,
        alpha=3e-3,
        dropout=0.10,
        validation_fraction=0.0,
        n_iter_no_change=30,
    ),
    config(
        "small_a1e-2_d00_b512_mean_centered_sorted_fulltrain",
        layers=SMALL_LAYERS,
        alpha=1e-2,
        dropout=0.0,
        validation_fraction=0.0,
        n_iter_no_change=30,
        vector_transform="mean_centered_sorted",
    ),
    config(
        "small_a7p5e-3_d00_b512_mean_centered_sorted_fulltrain",
        layers=SMALL_LAYERS,
        alpha=7.5e-3,
        dropout=0.0,
        validation_fraction=0.0,
        n_iter_no_change=30,
        vector_transform="mean_centered_sorted",
    ),
    config(
        "current_fulltrain",
        validation_fraction=0.0,
        n_iter_no_change=30,
    ),
    config(
        "compact_a1e-2_d10_b512_fulltrain",
        layers=COMPACT_LAYERS,
        alpha=1e-2,
        dropout=0.10,
        validation_fraction=0.0,
        n_iter_no_change=30,
    ),
    config(
        "current_a0_fulltrain",
        alpha=0.0,
        validation_fraction=0.0,
        n_iter_no_change=30,
    ),
    config(
        "current_a1e-6_fulltrain",
        alpha=1e-6,
        validation_fraction=0.0,
        n_iter_no_change=30,
    ),
    config(
        "current_a1e-5_fulltrain",
        alpha=1e-5,
        validation_fraction=0.0,
        n_iter_no_change=30,
    ),
    config(
        "current_a5e-5_fulltrain",
        alpha=5e-5,
        validation_fraction=0.0,
        n_iter_no_change=30,
    ),
    config(
        "current_a2e-4_fulltrain",
        alpha=2e-4,
        validation_fraction=0.0,
        n_iter_no_change=30,
    ),
    config(
        "current_a5e-4_fulltrain",
        alpha=5e-4,
        validation_fraction=0.0,
        n_iter_no_change=30,
    ),
    config(
        "current_d00_fulltrain",
        dropout=0.0,
        validation_fraction=0.0,
        n_iter_no_change=30,
    ),
    config(
        "current_d02_fulltrain",
        dropout=0.02,
        validation_fraction=0.0,
        n_iter_no_change=30,
    ),
    config(
        "current_d08_fulltrain",
        dropout=0.08,
        validation_fraction=0.0,
        n_iter_no_change=30,
    ),
    config(
        "current_d10_fulltrain",
        dropout=0.10,
        validation_fraction=0.0,
        n_iter_no_change=30,
    ),
    config(
        "current_b256_fulltrain",
        batch_size=256,
        validation_fraction=0.0,
        n_iter_no_change=30,
    ),
    config(
        "current_b1024_fulltrain",
        batch_size=1024,
        validation_fraction=0.0,
        n_iter_no_change=30,
    ),
    config(
        "current_lr1e-4_fulltrain",
        learning_rate=1e-4,
        validation_fraction=0.0,
        n_iter_no_change=30,
    ),
    config(
        "current_lr3e-4_fulltrain",
        learning_rate=3e-4,
        validation_fraction=0.0,
        n_iter_no_change=30,
    ),
    config(
        "medium_a1e-4_d05_fulltrain",
        layers=MEDIUM_LAYERS,
        validation_fraction=0.0,
        n_iter_no_change=30,
    ),
    config(
        "current_a1e-5_d02_fulltrain",
        alpha=1e-5,
        dropout=0.02,
        validation_fraction=0.0,
        n_iter_no_change=30,
    ),
    config(
        "current_a5e-5_d02_fulltrain",
        alpha=5e-5,
        dropout=0.02,
        validation_fraction=0.0,
        n_iter_no_change=30,
    ),
    # Low-weight-decay refinement around config 111.  All values stay within
    # the already screened full-training ranges for alpha, dropout, batch size,
    # and learning rate.  These configurations are screened with one seed
    # before any five-seed confirmation.
    config(
        "current_a1e-7_fulltrain",
        alpha=1e-7,
        validation_fraction=0.0,
        n_iter_no_change=30,
    ),
    config(
        "current_a3e-7_fulltrain",
        alpha=3e-7,
        validation_fraction=0.0,
        n_iter_no_change=30,
    ),
    config(
        "current_a3e-6_fulltrain",
        alpha=3e-6,
        validation_fraction=0.0,
        n_iter_no_change=30,
    ),
    config(
        "current_a5e-6_fulltrain",
        alpha=5e-6,
        validation_fraction=0.0,
        n_iter_no_change=30,
    ),
    config(
        "current_a7p5e-6_fulltrain",
        alpha=7.5e-6,
        validation_fraction=0.0,
        n_iter_no_change=30,
    ),
    config(
        "current_a1e-6_d02_fulltrain",
        alpha=1e-6,
        dropout=0.02,
        validation_fraction=0.0,
        n_iter_no_change=30,
    ),
    config(
        "current_a1e-6_d08_fulltrain",
        alpha=1e-6,
        dropout=0.08,
        validation_fraction=0.0,
        n_iter_no_change=30,
    ),
    config(
        "current_a1e-6_d10_fulltrain",
        alpha=1e-6,
        dropout=0.10,
        validation_fraction=0.0,
        n_iter_no_change=30,
    ),
    config(
        "current_a0_d02_fulltrain",
        alpha=0.0,
        dropout=0.02,
        validation_fraction=0.0,
        n_iter_no_change=30,
    ),
    config(
        "current_a0_d08_fulltrain",
        alpha=0.0,
        dropout=0.08,
        validation_fraction=0.0,
        n_iter_no_change=30,
    ),
    config(
        "current_a1e-6_b256_fulltrain",
        alpha=1e-6,
        batch_size=256,
        validation_fraction=0.0,
        n_iter_no_change=30,
    ),
    config(
        "current_a1e-6_b1024_fulltrain",
        alpha=1e-6,
        batch_size=1024,
        validation_fraction=0.0,
        n_iter_no_change=30,
    ),
    config(
        "current_a1e-6_lr1e-4_fulltrain",
        alpha=1e-6,
        learning_rate=1e-4,
        validation_fraction=0.0,
        n_iter_no_change=30,
    ),
    config(
        "current_a1e-6_lr3e-4_fulltrain",
        alpha=1e-6,
        learning_rate=3e-4,
        validation_fraction=0.0,
        n_iter_no_change=30,
    ),
    config(
        "current_a1e-6_d02_b256_fulltrain",
        alpha=1e-6,
        dropout=0.02,
        batch_size=256,
        validation_fraction=0.0,
        n_iter_no_change=30,
    ),
    config(
        "current_a1e-6_d02_lr1e-4_fulltrain",
        alpha=1e-6,
        dropout=0.02,
        learning_rate=1e-4,
        validation_fraction=0.0,
        n_iter_no_change=30,
    ),
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-id", type=int)
    parser.add_argument(
        "--variant",
        choices=("x_only", "x_one_llm", "one_logprob", "x_avg_llm", "x_all_llm"),
        default="x_all_llm",
    )
    parser.add_argument("--model-seed", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--eval-split", choices=("validation", "test"), default="validation")
    parser.add_argument("--outer-seed", type=int, default=20260807)
    parser.add_argument("--outer-validation-fraction", type=float, default=0.20)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--list-configs", action="store_true")
    return parser.parse_args()


def load_split(args):
    train = pd.read_parquet(TRAIN_PATH)
    if args.eval_split == "test":
        return train, pd.read_parquet(TEST_PATH)

    if not 0.0 < args.outer_validation_fraction < 1.0:
        raise ValueError("outer validation fraction must be between 0 and 1")
    rng = np.random.default_rng(args.outer_seed)
    order = rng.permutation(len(train))
    validation_size = max(1, int(len(train) * args.outer_validation_fraction))
    validation_index = order[:validation_size]
    subtrain_index = order[validation_size:]
    return train.iloc[subtrain_index].copy(), train.iloc[validation_index].copy()


def main():
    args = parse_args()
    if args.list_configs:
        print(json.dumps(CONFIGS, indent=2))
        return
    if args.config_id is None or args.output is None:
        raise ValueError("--config-id and --output are required unless --list-configs is used")
    if not 0 <= args.config_id < len(CONFIGS):
        raise ValueError(f"config id must be in [0, {len(CONFIGS) - 1}]")

    selected = CONFIGS[args.config_id]
    train, evaluation = load_split(args)
    train_features = prepare_features(
        train.to_dict("records"),
        args.variant,
        random_state=args.model_seed,
        llm_vector_transform=selected["vector_transform"],
    )
    eval_features = prepare_features(
        evaluation.to_dict("records"),
        args.variant,
        random_state=args.model_seed,
        llm_vector_transform=selected["vector_transform"],
    )
    x_train, y_train, *_ = train_features
    x_eval, y_eval, original_eval, ranges_eval, *_ = eval_features

    model = VariantDebiasModel(
        "mlp",
        device=args.device,
        random_state=args.model_seed,
        hidden_layers=selected["hidden_layers"],
        mlp_alpha=selected["alpha"],
        learning_rate_init=selected["learning_rate"],
        max_iter=3500,
        batch_size=selected["batch_size"],
        mlp_dropout=selected["dropout"],
        mlp_head="mse",
        validation_fraction=selected["validation_fraction"],
        n_iter_no_change=selected["n_iter_no_change"],
        min_delta=1e-6,
        mlp_standardize=selected["standardize"],
    )
    model.fit(x_train, y_train)
    prediction_norm = model.predict(x_eval)
    prediction_original = (
        prediction_norm * (ranges_eval[:, 1] - ranges_eval[:, 0]) + ranges_eval[:, 0]
    )

    mae_original = float(np.mean(np.abs(prediction_original - original_eval)))
    row = {
        "config_id": args.config_id,
        "config_name": selected["name"],
        "variant": args.variant,
        "eval_split": args.eval_split,
        "outer_seed": args.outer_seed,
        "model_seed": args.model_seed,
        "hidden_layers": selected["hidden_layers"],
        "alpha": selected["alpha"],
        "learning_rate": selected["learning_rate"],
        "dropout": selected["dropout"],
        "batch_size": selected["batch_size"],
        "validation_fraction": selected["validation_fraction"],
        "n_iter_no_change": selected["n_iter_no_change"],
        "standardize": selected["standardize"],
        "vector_transform": selected["vector_transform"],
        "train_rows": len(train),
        "eval_rows": len(evaluation),
        "mae": mae_original * 100.0,
        "acc": compute_accuracy_mad(original_eval, prediction_original, ranges_eval) * 100.0,
        "ha": compute_accuracy_hard(original_eval, prediction_original, ranges_eval) * 100.0,
        "sa": compute_accuracy_soft(original_eval, prediction_original, ranges_eval) * 100.0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(args.output, index=False, float_format="%.17g")
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
