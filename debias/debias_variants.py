"""
Debias algorithm variants for Average and Individual prediction

Supports two data formats:
1. Parquet format (Average prediction - OpinionQA/EEDI aggregated):
   - Question_Embedding: 256d
   - Average_LLM_Response_norm: Average LLM response (normalized)
   - LLM_Responses: All LLM responses list (normalized, e.g., 80 responses)

2. Parquet format (Individual prediction - OpinionQA individual):
   - Question_Embedding: 256d
   - Demographic_Embedding: 256d
   - LLM_Responses: All LLM responses list (normalized)
   - Human_Response_norm: Individual human response (normalized)

Algorithm variants:
1. x_only: Use only embedding features to predict human response
2. x_avg_llm: Use embeddings + average LLM response to predict human response
3. x_all_llm: Use embeddings + all LLM responses to predict human response
4. x_one_llm: Use embeddings + first LLM response to predict human response
5. one_logprob: Use embeddings + a fixed-width answer probability vector
"""
import json
import argparse
import os
import copy
import hashlib
import math
from types import SimpleNamespace
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    xgb = None
    XGBOOST_AVAILABLE = False

# Try to import PyTorch for MLP with GPU support
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = SimpleNamespace(Module=object)
    optim = None
    DataLoader = None
    TensorDataset = None
    print("Warning: PyTorch not available. MLP model will fall back to sklearn MLPRegressor (no GPU support).")

# Import evaluation functions
from debias.evaluate_variants import (
    compute_metrics,
    compute_accuracy_hard,
    compute_accuracy_soft,
    compute_accuracy_mad,
    print_experiment_results,
    save_single_result_to_csv
)


ONE_LOGPROB_VARIANT = "one_logprob"
ONE_LOGPROB_FIELD = "one_logprobs"
ONE_LOGPROB_BASELINE_FIELD = "one_logprobs_expected_response_norm"
ONE_LOGPROB_CHOICE_COUNT_FIELD = "one_logprobs_choice_count"


def load_data(file_path):
    """
    Load data from JSON or parquet file

    Args:
        file_path: Path to data file (JSON or parquet)

    Returns:
        List of dictionaries (same format for both JSON and parquet)
    """
    file_path = Path(file_path)

    if file_path.suffix == '.parquet':
        # Load from parquet
        print(f"Loading parquet file: {file_path}")
        df = pd.read_parquet(file_path)
        # Convert DataFrame rows to list of dicts
        data = df.to_dict('records')
        print(f"Loaded {len(data)} samples from parquet")
        return data
    elif file_path.suffix == '.json':
        # Load from JSON
        print(f"Loading JSON file: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Loaded {len(data)} samples from JSON")
        return data
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}. Use .json or .parquet")


def is_individual_data(data):
    """
    Detect if data is individual format (has Demographic_Embedding)

    Args:
        data: List of data samples

    Returns:
        True if individual format, False if average format
    """
    if len(data) == 0:
        return False
    # Check first sample for Demographic_Embedding
    first_sample = data[0]
    return 'Demographic_Embedding' in first_sample and len(first_sample.get('Demographic_Embedding', [])) > 0


def _to_llm_response_list(raw_value):
    """
    Convert a raw LLM response field to a Python list.

    Supports list/tuple/ndarray directly; scalar values are wrapped into a
    single-element list for robustness.
    """
    if raw_value is None:
        return []
    if isinstance(raw_value, np.ndarray):
        return raw_value.tolist()
    if isinstance(raw_value, (list, tuple)):
        return list(raw_value)
    return [raw_value]


def _one_logprob_score_range(item, is_individual):
    """Return and strictly validate the integer response scale for one row."""
    if is_individual:
        raw_min = item.get("score_range_min")
        raw_max = item.get("score_range_max")
    else:
        raw_range = item.get("score_range")
        if raw_range is None or len(raw_range) < 2:
            raise ValueError("one_logprob requires score_range on population rows")
        raw_min, raw_max = raw_range[0], raw_range[-1]

    try:
        score_min = float(raw_min)
        score_max = float(raw_max)
    except (TypeError, ValueError) as exc:
        raise ValueError("one_logprob requires a numeric response scale") from exc
    if (
        not math.isfinite(score_min)
        or not math.isfinite(score_max)
        or not score_min.is_integer()
        or not score_max.is_integer()
        or score_min >= score_max
    ):
        raise ValueError(
            f"one_logprob received invalid response scale [{raw_min}, {raw_max}]"
        )
    return int(score_min), int(score_max)


def _prepare_one_logprob_block(
    item,
    *,
    is_individual,
    feature_field=ONE_LOGPROB_FIELD,
):
    """Validate one materialized PMF and return it with its expected baseline."""
    missing = [
        field
        for field in (
            feature_field,
            ONE_LOGPROB_BASELINE_FIELD,
            ONE_LOGPROB_CHOICE_COUNT_FIELD,
        )
        if field not in item
    ]
    if missing:
        raise ValueError(f"one_logprob row is missing required fields: {missing}")

    try:
        probabilities = np.asarray(item[feature_field], dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"one_logprob field {feature_field!r} must be a numeric vector"
        ) from exc
    if probabilities.ndim != 1 or len(probabilities) == 0:
        raise ValueError(
            f"one_logprob field {feature_field!r} must be a non-empty vector"
        )
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0.0):
        raise ValueError("one_logprob probabilities must be finite and nonnegative")
    if not math.isclose(
        math.fsum(probabilities.tolist()), 1.0, rel_tol=0.0, abs_tol=1e-8
    ):
        raise ValueError("one_logprob probabilities must sum to one")

    raw_choice_count = item[ONE_LOGPROB_CHOICE_COUNT_FIELD]
    try:
        choice_count_float = float(raw_choice_count)
        choice_count = int(choice_count_float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("one_logprob choice count must be an integer") from exc
    if (
        not math.isfinite(choice_count_float)
        or choice_count_float != choice_count
        or choice_count <= 0
        or choice_count > len(probabilities)
    ):
        raise ValueError(
            f"one_logprob received invalid choice count {raw_choice_count!r}"
        )

    score_min, score_max = _one_logprob_score_range(item, is_individual)
    expected_choice_count = score_max - score_min + 1
    if choice_count != expected_choice_count:
        raise ValueError(
            f"one_logprob choice count {choice_count} does not match response "
            f"scale [{score_min}, {score_max}]"
        )
    if not np.allclose(
        probabilities[choice_count:], 0.0, rtol=0.0, atol=1e-12
    ):
        raise ValueError("one_logprob padded probability slots must be zero")

    try:
        baseline = float(item[ONE_LOGPROB_BASELINE_FIELD])
    except (TypeError, ValueError) as exc:
        raise ValueError("one_logprob expected-response baseline must be numeric") from exc
    if not math.isfinite(baseline) or not -1e-10 <= baseline <= 1.0 + 1e-10:
        raise ValueError("one_logprob expected-response baseline is out of range")

    scores = np.arange(score_min, score_max + 1, dtype=float)
    expected_score = float(np.dot(scores, probabilities[:choice_count]))
    expected_norm = (expected_score - score_min) / (score_max - score_min)
    if not math.isclose(baseline, expected_norm, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError(
            "one_logprob expected-response baseline does not match its probabilities"
        )
    return probabilities, float(np.clip(baseline, 0.0, 1.0))


DEFAULT_SAMPLE_LLM_FIELDS = [
    'claude-3.5-haiku_norm',
    'deepseek-v3_norm',
    'gpt-3.5-turbo_norm',
    'gpt-4o-mini_norm',
    'gpt-4o_norm',
    'gpt-5-mini_norm',
    'llama-3.3-70B-instruct-turbo_norm',
    'mistral-7B-instruct-v0.3_norm',
]

VALID_LLM_VECTOR_TRANSFORMS = [
    "raw",
    "sorted",
    "centered_sorted",
    "mean_centered_sorted",
]


def _normalize_llm_fields(llm_fields):
    """Normalize a comma-separated or iterable LLM field specification."""
    if llm_fields is None:
        return None

    if isinstance(llm_fields, str):
        fields = [field.strip() for field in llm_fields.split(",")]
    else:
        fields = [str(field).strip() for field in llm_fields]

    fields = [field for field in fields if field]
    return fields or None


def _get_item_llm_responses(item, llm_field=None, llm_fields=None):
    """Fetch LLM responses from one or more fields, or the default field."""
    normalized_llm_fields = _normalize_llm_fields(llm_fields)
    if llm_field and normalized_llm_fields:
        raise ValueError("Use either llm_field or llm_fields, not both.")

    if normalized_llm_fields:
        merged = []
        for field_name in normalized_llm_fields:
            if field_name not in item:
                raise KeyError(
                    f"LLM field '{field_name}' not found in sample. "
                    f"Available keys include: {list(item.keys())[:20]}"
                )
            merged.extend(_to_llm_response_list(item.get(field_name, [])))
        return merged

    if llm_field:
        if llm_field not in item:
            raise KeyError(
                f"LLM field '{llm_field}' not found in sample. "
                f"Available keys include: {list(item.keys())[:20]}"
            )
        raw = item.get(llm_field, [])
    else:
        raw = item.get('LLM_Responses_norm', [])
    return _to_llm_response_list(raw)


def _resolve_item_llm_fields(llm_field=None, llm_fields=None, allow_default_sample_fields=False):
    normalized_llm_fields = _normalize_llm_fields(llm_fields)
    if llm_field and normalized_llm_fields:
        raise ValueError("Use either llm_field or llm_fields, not both.")
    if normalized_llm_fields:
        return normalized_llm_fields
    if llm_field:
        return [llm_field]
    if allow_default_sample_fields:
        return list(DEFAULT_SAMPLE_LLM_FIELDS)
    return None


def _stable_item_identifier(item):
    candidate_keys = [
        "Variable_Name",
        "question_id",
        "Question",
        "ParticipantID",
        "simulation_id",
    ]
    parts = []
    for key in candidate_keys:
        value = item.get(key)
        if value is not None:
            parts.append(f"{key}={value}")
    if parts:
        return "|".join(parts)
    return json.dumps(sorted(item.keys()))


def _sample_llm_responses(responses, sample_per_llm, rng):
    if sample_per_llm <= 0:
        raise ValueError(f"sample_per_llm must be positive, got {sample_per_llm}.")

    numeric_responses = [float(value) for value in responses]
    if len(numeric_responses) >= sample_per_llm:
        sampled_indices = rng.choice(len(numeric_responses), size=sample_per_llm, replace=False)
        return [numeric_responses[int(idx)] for idx in sampled_indices.tolist()]
    if len(numeric_responses) > 0:
        permuted_indices = rng.permutation(len(numeric_responses))
        sampled = [numeric_responses[int(idx)] for idx in permuted_indices.tolist()]
        mean_val = float(np.mean(numeric_responses))
        sampled.extend([mean_val] * (sample_per_llm - len(sampled)))
        return sampled
    return [0.0] * sample_per_llm


def _transform_llm_response_block(responses, llm_vector_transform="raw"):
    """Apply an order-invariant transform to one response block."""
    numeric_responses = [float(value) for value in responses]

    if llm_vector_transform == "raw":
        return numeric_responses
    if llm_vector_transform == "sorted":
        return sorted(numeric_responses)
    if llm_vector_transform == "centered_sorted":
        if len(numeric_responses) == 0:
            return []
        mean_val = float(np.mean(numeric_responses))
        return sorted(value - mean_val for value in numeric_responses)
    if llm_vector_transform == "mean_centered_sorted":
        if len(numeric_responses) == 0:
            return []
        mean_val = float(np.mean(numeric_responses))
        centered = sorted(value - mean_val for value in numeric_responses)
        # Final length is handled in _build_llm_vector_from_blocks.
        return [mean_val] + centered

    raise ValueError(
        f"Unsupported llm_vector_transform={llm_vector_transform}. "
        f"Choose from: {', '.join(VALID_LLM_VECTOR_TRANSFORMS)}"
    )


def _build_processed_llm_response_blocks(
    item,
    llm_field=None,
    llm_fields=None,
    llm_sampling_strategy="all",
    sample_per_llm=10,
    random_state=42,
    allow_default_sample_fields=False,
):
    """Return one processed response block per field before vector transforms."""
    resolved_fields = _resolve_item_llm_fields(
        llm_field=llm_field,
        llm_fields=llm_fields,
        allow_default_sample_fields=allow_default_sample_fields,
    )

    if resolved_fields is None:
        return [_get_item_llm_responses(item, llm_field=None, llm_fields=None)]

    if llm_sampling_strategy == "all":
        return [_to_llm_response_list(item.get(field_name, [])) for field_name in resolved_fields]

    if llm_sampling_strategy != "sample_per_field":
        raise ValueError(f"Unsupported llm_sampling_strategy: {llm_sampling_strategy}")

    item_identifier = _stable_item_identifier(item)
    processed_blocks = []
    for field_name in resolved_fields:
        responses = _to_llm_response_list(item.get(field_name, []))
        seed_material = f"{random_state}|{field_name}|{item_identifier}"
        seed = int.from_bytes(hashlib.sha256(seed_material.encode("utf-8")).digest()[:8], "little")
        rng = np.random.default_rng(seed)
        processed_blocks.append(_sample_llm_responses(responses, sample_per_llm, rng))
    return processed_blocks


def _flatten_llm_response_blocks(blocks, llm_vector_transform="raw"):
    flattened = []
    for block in blocks:
        flattened.extend(
            _transform_llm_response_block(block, llm_vector_transform=llm_vector_transform)
        )
    return flattened


def _pad_value_for_vector_transform(raw_llm_responses, llm_vector_transform):
    if llm_vector_transform == "centered_sorted":
        return 0.0
    if llm_vector_transform == "mean_centered_sorted":
        return 0.0
    if len(raw_llm_responses) == 0:
        return 0.0
    return float(np.mean(raw_llm_responses))


def _compress_sorted_values(sorted_values, target_dim):
    """Compress a sorted sequence to fixed length via evenly spaced order stats."""
    if target_dim <= 0:
        return []
    values = np.asarray(sorted_values, dtype=float)
    if len(values) == 0:
        return [0.0] * target_dim
    if len(values) == target_dim:
        return values.tolist()
    if len(values) < target_dim:
        pad_value = float(values[-1])
        return values.tolist() + [pad_value] * (target_dim - len(values))

    positions = np.linspace(0, len(values) - 1, num=target_dim)
    lower = np.floor(positions).astype(int)
    upper = np.ceil(positions).astype(int)
    weight = positions - lower
    compressed = values[lower] * (1.0 - weight) + values[upper] * weight
    return compressed.tolist()


def _apply_llm_dim(llm_responses, llm_dim, pad_value):
    used_llm = list(llm_responses)
    if llm_dim is None:
        return used_llm
    if len(used_llm) > llm_dim:
        return used_llm[:llm_dim]
    if len(used_llm) < llm_dim:
        return used_llm + [pad_value] * (llm_dim - len(used_llm))
    return used_llm


def _build_llm_vector_from_blocks(blocks, llm_dim=None, llm_vector_transform="raw"):
    """Build the final vector-valued representation from processed response blocks."""
    if llm_vector_transform in {"raw", "sorted", "centered_sorted"}:
        transformed = _flatten_llm_response_blocks(blocks, llm_vector_transform=llm_vector_transform)
        if llm_vector_transform in {"sorted", "centered_sorted"} and llm_dim is not None and len(transformed) > llm_dim:
            return _compress_sorted_values(transformed, llm_dim)
        return _apply_llm_dim(
            transformed,
            llm_dim,
            pad_value=_pad_value_for_vector_transform(
                transformed,
                llm_vector_transform=llm_vector_transform,
            ),
        )

    if llm_vector_transform == "mean_centered_sorted":
        per_block_vectors = []
        for block in blocks:
            numeric_block = [float(value) for value in block]
            if len(numeric_block) == 0:
                per_block_vectors.append([])
                continue
            mean_val = float(np.mean(numeric_block))
            residuals = sorted(value - mean_val for value in numeric_block)
            block_target_dim = len(numeric_block) if llm_dim is None else llm_dim
            residual_target_dim = max(block_target_dim - 1, 0)
            compressed_residuals = _compress_sorted_values(residuals, residual_target_dim)
            per_block_vectors.append([mean_val] + compressed_residuals)
        return [value for block_vec in per_block_vectors for value in block_vec]

    raise ValueError(
        f"Unsupported llm_vector_transform={llm_vector_transform}. "
        f"Choose from: {', '.join(VALID_LLM_VECTOR_TRANSFORMS)}"
    )


def _build_processed_llm_responses(
    item,
    llm_field=None,
    llm_fields=None,
    llm_sampling_strategy="all",
    sample_per_llm=10,
    random_state=42,
    allow_default_sample_fields=False,
    llm_vector_transform="raw",
):
    blocks = _build_processed_llm_response_blocks(
        item,
        llm_field=llm_field,
        llm_fields=llm_fields,
        llm_sampling_strategy=llm_sampling_strategy,
        sample_per_llm=sample_per_llm,
        random_state=random_state,
        allow_default_sample_fields=allow_default_sample_fields,
    )
    return _flatten_llm_response_blocks(blocks, llm_vector_transform=llm_vector_transform)


def get_max_llm_response_length(
    data,
    llm_field=None,
    llm_fields=None,
    llm_sampling_strategy="all",
    sample_per_llm=10,
    random_state=42,
):
    """
    Get maximum available LLM response length in dataset.

    Returns:
        max_len: Maximum length across non-empty response lists.
    """
    max_len = 0
    for item in data:
        responses = _build_processed_llm_responses(
            item,
            llm_field=llm_field,
            llm_fields=llm_fields,
            llm_sampling_strategy=llm_sampling_strategy,
            sample_per_llm=sample_per_llm,
            random_state=random_state,
            allow_default_sample_fields=(llm_sampling_strategy == "sample_per_field"),
        )
        if len(responses) > max_len:
            max_len = len(responses)
    return max_len


def prepare_features(
    data,
    variant,
    llm_field=None,
    llm_fields=None,
    sample_per_llm=10,
    llm_dim=None,
    llm_sampling_strategy="all",
    random_state=42,
    llm_vector_transform="raw",
):
    """
    Prepare features based on variant

    Variant options:
    - 'x_only': Embeddings only
      - Individual: Demographic_Embedding + Question_Embedding (512d)
      - Average: Question_Embedding only (256d)
    - 'x_avg_llm': Embeddings + Average_LLM_Response_norm (513d or 513d)
    - 'x_all_llm': Embeddings + All_LLM_Responses (base + n)
    - 'x_one_llm': Embeddings + First_LLM_Response (513d)
    - 'one_logprob': Question embedding + optional demographic embedding +
      fixed-width answer probabilities. The individual feature order is kept
      compatible with the original One Logprob benchmark.

    Args:
        data: List of data samples
        variant: Feature variant to use
        llm_field: Optional custom LLM response field (e.g., 'claude-3.5-haiku_norm')
                   If specified, will use this field instead of 'LLM_Responses_norm'
        llm_fields: Optional comma-separated or iterable list of LLM fields.
                    When specified, multi-source variants concatenate in the listed order.
        sample_per_llm: Responses sampled per field when using sample_per_field
        llm_dim: Target LLM response dimension for x_all_llm.
                 If provided, each sample is truncated/padded to this length.
        llm_sampling_strategy: "all" or "sample_per_field"
        random_state: Random seed used by per-field sampling
        llm_vector_transform: How to canonicalize vector-valued LLM inputs.

    Returns: X, y, original_responses, score_ranges, baseline_llm
    """
    # Detect data format
    is_individual = is_individual_data(data)

    X = []
    y = []
    original_responses = []
    score_ranges = []
    baseline_llm = []
    twin_ids = []

    normalized_llm_fields = _normalize_llm_fields(llm_fields)
    use_processed_custom_llm = bool(llm_field or normalized_llm_fields or llm_sampling_strategy == "sample_per_field")

    for item in data:
        # Get embeddings
        question_embedding = np.array(item.get('Question_Embedding', []))

        # Skip items without embedding
        if len(question_embedding) == 0:
            continue

        # Get demographic embedding (only for individual format)
        if is_individual:
            demographic_embedding = np.array(item.get('Demographic_Embedding', []))
            if len(demographic_embedding) == 0:
                continue
            if variant == ONE_LOGPROB_VARIANT:
                # Preserve the exact feature order of the isolated benchmark.
                base_features = np.concatenate(
                    [question_embedding, demographic_embedding]
                )
            else:
                # Existing variants retain their historical feature order.
                base_features = np.concatenate(
                    [demographic_embedding, question_embedding]
                )
        else:
            # Average format: only question embedding
            base_features = question_embedding.copy()

        features = base_features.copy()

        # Calculate baseline value (will be set in variant branches)
        baseline_value = None

        # Add extra features based on variant
        if variant == 'x_avg_llm':
            # Use average LLM response (normalized)
            avg_llm = item.get('Average_LLM_Response_norm')
            if not is_individual and use_processed_custom_llm:
                raw_llm_responses = _build_processed_llm_responses(
                    item,
                    llm_field=llm_field,
                    llm_fields=normalized_llm_fields,
                    llm_sampling_strategy=llm_sampling_strategy,
                    sample_per_llm=sample_per_llm,
                    random_state=random_state,
                    llm_vector_transform="raw",
                )
                if len(raw_llm_responses) > 0:
                    avg_llm = float(np.mean(raw_llm_responses))
            if is_individual:
                # Individual format: use Average_LLM_Response_norm
                avg_llm = item.get('Average_LLM_Response_norm', 0)
            if avg_llm is None:
                avg_llm = 0
            features = np.append(features, avg_llm)
            baseline_value = avg_llm

        elif variant == 'x_all_llm':
            # Use all LLM responses
            raw_blocks = _build_processed_llm_response_blocks(
                item,
                llm_field=llm_field,
                llm_fields=normalized_llm_fields,
                llm_sampling_strategy=llm_sampling_strategy,
                sample_per_llm=sample_per_llm,
                random_state=random_state,
            )
            raw_llm_responses = [value for block in raw_blocks for value in block]
            llm_responses = _build_llm_vector_from_blocks(
                raw_blocks,
                llm_dim=llm_dim,
                llm_vector_transform=llm_vector_transform,
            )

            if len(llm_responses) == 0:
                continue  # Skip this sample entirely

            llm_array = np.array(llm_responses, dtype=float)
            features = np.concatenate([features, llm_array])
            baseline_value = float(np.mean(raw_llm_responses)) if len(raw_llm_responses) > 0 else float(np.mean(llm_array))

        elif variant == 'x_one_llm':
            # Use only first LLM response
            llm_responses = _build_processed_llm_responses(
                item,
                llm_field=llm_field,
                llm_fields=normalized_llm_fields,
                llm_sampling_strategy=llm_sampling_strategy,
                sample_per_llm=sample_per_llm,
                random_state=random_state,
            )

            if len(llm_responses) == 0:
                continue  # Skip this sample entirely
            features = np.append(features, llm_responses[0])
            baseline_value = llm_responses[0]

        elif variant == ONE_LOGPROB_VARIANT:
            feature_field = llm_field or ONE_LOGPROB_FIELD
            probability_vector, baseline_value = _prepare_one_logprob_block(
                item,
                is_individual=is_individual,
                feature_field=feature_field,
            )
            features = np.concatenate([features, probability_vector])

        elif variant == 'x_only':
            # Only use embeddings. When a multi-source field set is supplied,
            # use its shared mean-pooled signal as the baseline comparator.
            if is_individual:
                baseline_value = item.get('Average_LLM_Response_norm', 0)
            else:
                if use_processed_custom_llm:
                    # Use custom source(s) for baseline.
                    llm_responses = _build_processed_llm_responses(
                        item,
                        llm_field=llm_field,
                        llm_fields=normalized_llm_fields,
                        llm_sampling_strategy=llm_sampling_strategy,
                        sample_per_llm=sample_per_llm,
                        random_state=random_state,
                    )
                    if len(llm_responses) > 0:
                        if normalized_llm_fields:
                            baseline_value = float(np.mean(llm_responses))
                        else:
                            baseline_value = llm_responses[0]
                    else:
                        baseline_value = item.get('Average_LLM_Response_norm', 0)
                        if baseline_value is None:
                            baseline_value = 0
                else:
                    # Shared no-LLM baseline should use the averaged LLM signal.
                    baseline_value = item.get('Average_LLM_Response_norm', 0)
                    if baseline_value is None:
                        llm_responses = _get_item_llm_responses(item, llm_field=None)
                        if len(llm_responses) > 0:
                            baseline_value = llm_responses[0]
                        else:
                            baseline_value = 0

        # Only append if we successfully calculated baseline_value
        if baseline_value is not None:
            X.append(features)
            baseline_llm.append(float(baseline_value))

            # Target is normalized human response
            if is_individual:
                y.append(item.get('Human_Response_norm', 0))
                original_responses.append(item.get('Human_Response', 0))
                # Get score range
                score_min = item.get('score_range_min', 1)
                score_max = item.get('score_range_max', 5)
                score_ranges.append([score_min, score_max])
                twin_ids.append(item.get('twin_id'))
            else:
                y.append(item.get('Average_Human_Response_norm', 0))
                original_responses.append(item.get('Average_Human_Response', 0))
                score_ranges.append(item.get('score_range', [1, 9]))
                twin_ids.append(None)

    # Debug: check array lengths before conversion
    print(f"Debug: X={len(X)}, y={len(y)}, orig={len(original_responses)}, ranges={len(score_ranges)}, baseline={len(baseline_llm)}")

    return (
        np.array(X),
        np.array(y),
        np.array(original_responses),
        np.array(score_ranges),
        np.array(baseline_llm),
        np.array(twin_ids, dtype=object),
    )


class PyTorchMLP(nn.Module):
    """PyTorch MLP with GPU support"""

    def __init__(self, input_dim, hidden_layers, dropout=0.0):
        super().__init__()
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Output layer (single regression output)
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze()


class PyTorchMLPRegressor:
    """Wrapper for PyTorch MLP to match sklearn-like API with GPU support"""

    def __init__(self, hidden_layers=(2048, 1024, 512, 256),
                 learning_rate=0.0005, alpha=0.00001, batch_size=512,
                 max_iter=3000, early_stopping=True, n_iter_no_change=30,
                 validation_fraction=0.1, min_delta=1e-6,
                 dropout=0.0, standardize=True,
                 lr_patience=10, lr_factor=0.5, min_lr=1e-6,
                 device='cuda', random_state=42, verbose=False):
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        self.n_iter_no_change = n_iter_no_change
        self.validation_fraction = validation_fraction
        self.min_delta = min_delta
        self.dropout = dropout
        self.standardize = standardize
        self.lr_patience = lr_patience
        self.lr_factor = lr_factor
        self.min_lr = min_lr
        self.random_state = random_state
        self.verbose = verbose

        # Set device - supports 'cuda', 'cuda:0', 'cuda:1', 'cpu'
        if device == 'cuda' and TORCH_AVAILABLE and torch.cuda.is_available():
            # Default to cuda:0 if just 'cuda' is specified
            self.device = torch.device('cuda:0')
            if verbose:
                print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        elif device.startswith('cuda:') and TORCH_AVAILABLE:
            # Specific GPU device (e.g., 'cuda:0', 'cuda:1')
            device_id = int(device.split(':')[1])
            if device_id < torch.cuda.device_count():
                self.device = torch.device(device)
                if verbose:
                    print(f"Using GPU {device}: {torch.cuda.get_device_name(device_id)}")
            else:
                self.device = torch.device('cpu')
                if verbose:
                    print(f"GPU {device} not available (total GPUs: {torch.cuda.device_count()}), using CPU")
        elif device == 'cuda' and TORCH_AVAILABLE and not torch.cuda.is_available():
            self.device = torch.device('cpu')
            if verbose:
                print("CUDA requested but not available, using CPU")
        else:
            self.device = torch.device('cpu')
            if verbose:
                print("Using CPU")

        self.model = None
        self.input_dim = None
        self.feature_mean = None
        self.feature_std = None

    def _create_model(self, input_dim):
        self.input_dim = input_dim
        model = PyTorchMLP(input_dim, self.hidden_layers, dropout=self.dropout)
        return model.to(self.device)

    def fit(self, X, y):
        # Set random seed
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_state)

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        if self.standardize:
            self.feature_mean = X.mean(axis=0)
            self.feature_std = X.std(axis=0)
            self.feature_std[self.feature_std < 1e-8] = 1.0
            X_proc = (X - self.feature_mean) / self.feature_std
        else:
            self.feature_mean = None
            self.feature_std = None
            X_proc = X

        n_samples = X_proc.shape[0]
        use_validation = self.early_stopping and self.validation_fraction > 0 and n_samples >= 10

        if use_validation:
            rng = np.random.default_rng(self.random_state)
            indices = rng.permutation(n_samples)
            val_size = max(1, int(n_samples * self.validation_fraction))
            val_idx = indices[:val_size]
            train_idx = indices[val_size:]
            if train_idx.size == 0:
                train_idx = val_idx
                val_idx = np.array([], dtype=np.int64)
                use_validation = False
        else:
            train_idx = np.arange(n_samples, dtype=np.int64)
            val_idx = np.array([], dtype=np.int64)

        # Convert to tensors
        X_train_tensor = torch.as_tensor(X_proc[train_idx], dtype=torch.float32, device=self.device)
        y_train_tensor = torch.as_tensor(y[train_idx], dtype=torch.float32, device=self.device)
        if use_validation:
            X_val_tensor = torch.as_tensor(X_proc[val_idx], dtype=torch.float32, device=self.device)
            y_val_tensor = torch.as_tensor(y[val_idx], dtype=torch.float32, device=self.device)

        # Create model
        self.model = self._create_model(X_proc.shape[1])

        # Loss and optimizer
        criterion = nn.MSELoss()
        # Add L2 regularization through weight_decay
        optimizer = optim.Adam(self.model.parameters(),
                              lr=self.learning_rate,
                              weight_decay=self.alpha)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=self.lr_factor,
            patience=self.lr_patience,
            min_lr=self.min_lr
        )

        # Data loader
        dataset = TensorDataset(X_train_tensor, y_train_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Training loop
        best_loss = float('inf')
        patience_counter = 0
        best_model_state = copy.deepcopy(self.model.state_dict())

        for epoch in range(self.max_iter):
            self.model.train()
            epoch_loss = 0.0

            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                predictions = self.model(batch_X)
                loss = criterion(predictions, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)
            monitor_loss = avg_loss

            if use_validation:
                self.model.eval()
                with torch.no_grad():
                    val_predictions = self.model(X_val_tensor)
                    val_loss = criterion(val_predictions, y_val_tensor).item()
                monitor_loss = val_loss
            else:
                val_loss = None

            scheduler.step(monitor_loss)

            # Early stopping
            if self.early_stopping:
                if monitor_loss < best_loss - self.min_delta:
                    best_loss = monitor_loss
                    patience_counter = 0
                    # Save best model
                    best_model_state = copy.deepcopy(self.model.state_dict())
                else:
                    patience_counter += 1
                    if patience_counter >= self.n_iter_no_change:
                        if self.verbose:
                            print(f"Early stopping at epoch {epoch} (best_loss={best_loss:.6f})")
                        # Restore best model
                        self.model.load_state_dict(best_model_state)
                        break

            if self.verbose and epoch % 100 == 0:
                lr = optimizer.param_groups[0]['lr']
                if val_loss is None:
                    print(f"Epoch {epoch}, Train Loss: {avg_loss:.6f}, LR: {lr:.6g}")
                else:
                    print(f"Epoch {epoch}, Train Loss: {avg_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {lr:.6g}")

        if self.early_stopping:
            self.model.load_state_dict(best_model_state)

        return self

    def predict(self, X):
        self.model.eval()
        X = np.asarray(X, dtype=np.float32)
        if self.standardize and self.feature_mean is not None and self.feature_std is not None:
            X = (X - self.feature_mean) / self.feature_std
        # Chunked forward pass so peak memory is bounded regardless of test
        # set size; avoids OOM on Twin-2K-500's 33k-row test split when many
        # concurrent configs contend for the same GPU.
        chunk = 2048
        outs = []
        with torch.no_grad():
            for start in range(0, X.shape[0], chunk):
                end = min(start + chunk, X.shape[0])
                X_tensor = torch.as_tensor(
                    X[start:end], dtype=torch.float32, device=self.device
                )
                preds = self.model(X_tensor)
                outs.append(preds.detach().cpu())
                del X_tensor, preds
        return torch.cat(outs, dim=0).numpy()


class VariantDebiasModel:
    """Debias model with different feature combinations and GPU support"""

    def __init__(self, model_type='ridge', device='cuda', random_state=42, **kwargs):
        """
        model_type options:
        - 'ridge': Ridge regression
        - 'rf': Random forest
        - 'mlp': Multi-layer perceptron (uses PyTorch with GPU support if available)
        - 'xgboost': XGBoost regression (with GPU support)

        device options:
        - 'cuda': Use GPU if available (defaults to cuda:0)
        - 'cuda:0', 'cuda:1', etc.: Use specific GPU device
        - 'cpu': Use CPU
        """
        self.model_type = model_type
        self.device = device
        self.random_state = random_state
        self.kwargs = kwargs
        self.model = None
        self.feature_dim = None

    def _create_model(self, input_dim):
        if self.model_type == 'ridge':
            return Ridge(alpha=self.kwargs.get('alpha', 1.0))
        elif self.model_type == 'rf':
            return RandomForestRegressor(
                n_estimators=self.kwargs.get('n_estimators', 100),
                max_depth=self.kwargs.get('max_depth', 10),
                random_state=self.random_state,
                n_jobs=-1
            )
        elif self.model_type == 'mlp':
            # Parse hidden_layer_sizes from string like "2048,1024,512,256"
            hidden_layers = self.kwargs.get('hidden_layers', '2048,1024,512,256')
            if isinstance(hidden_layers, str):
                hidden_layer_sizes = tuple(map(int, hidden_layers.split(',')))
            else:
                hidden_layer_sizes = hidden_layers

            # Use PyTorch MLP only when the requested CUDA device is actually available.
            # On some macOS CPU environments, PyTorch MLP may fail when it silently falls
            # back from an unavailable CUDA device to CPU, so prefer sklearn in that case.
            cuda_requested = TORCH_AVAILABLE and str(self.device).startswith('cuda')
            cuda_available_for_request = False
            if cuda_requested:
                if str(self.device) == 'cuda':
                    cuda_available_for_request = torch.cuda.is_available()
                elif str(self.device).startswith('cuda:'):
                    try:
                        device_id = int(str(self.device).split(':')[1])
                        cuda_available_for_request = (
                            torch.cuda.is_available() and device_id < torch.cuda.device_count()
                        )
                    except (IndexError, ValueError):
                        cuda_available_for_request = False

            use_torch_mlp = cuda_requested and cuda_available_for_request
            if use_torch_mlp:
                # Always enable verbose for PyTorch MLP to show device info
                return PyTorchMLPRegressor(
                    hidden_layers=hidden_layer_sizes,
                    learning_rate=self.kwargs.get('learning_rate_init', 0.0005),
                    alpha=self.kwargs.get('mlp_alpha', 0.00001),
                    batch_size=self.kwargs.get('batch_size', 512),
                    max_iter=self.kwargs.get('max_iter', 3000),
                    n_iter_no_change=self.kwargs.get('n_iter_no_change', 30),
                    validation_fraction=self.kwargs.get('validation_fraction', 0.1),
                    min_delta=self.kwargs.get('min_delta', 1e-6),
                    dropout=self.kwargs.get('mlp_dropout', 0.0),
                    standardize=self.kwargs.get('mlp_standardize', True),
                    device=self.device,
                    random_state=self.random_state,
                    verbose=True  # Always show device info
                )
            else:
                if cuda_requested and not cuda_available_for_request:
                    print("MLP CUDA requested but unavailable; using sklearn MLPRegressor on CPU.")
                elif TORCH_AVAILABLE and str(self.device) == 'cpu':
                    print("MLP on CPU: using sklearn MLPRegressor for stability.")
                from sklearn.neural_network import MLPRegressor
                sklearn_mlp = MLPRegressor(
                    hidden_layer_sizes=hidden_layer_sizes,
                    alpha=self.kwargs.get('mlp_alpha', 0.00001),
                    learning_rate='adaptive',
                    learning_rate_init=self.kwargs.get('learning_rate_init', 0.0005),
                    max_iter=self.kwargs.get('max_iter', 3000),
                    early_stopping=True,
                    validation_fraction=self.kwargs.get('validation_fraction', 0.1),
                    n_iter_no_change=self.kwargs.get('n_iter_no_change', 30),
                    batch_size=self.kwargs.get('batch_size', 512),
                    random_state=self.random_state,
                    verbose=False
                )
                if self.kwargs.get('mlp_standardize', True):
                    return make_pipeline(StandardScaler(), sklearn_mlp)
                return sklearn_mlp
        elif self.model_type == 'xgboost':
            if not XGBOOST_AVAILABLE:
                raise ImportError(
                    "xgboost is not installed. Install xgboost or choose a different model_type."
                )
            # XGBoost with GPU support
            xgb_params = {
                'n_estimators': self.kwargs.get('n_estimators', 100),
                'max_depth': self.kwargs.get('max_depth', 6),
                'learning_rate': self.kwargs.get('learning_rate', 0.1),
                'subsample': self.kwargs.get('subsample', 0.8),
                'colsample_bytree': self.kwargs.get('colsample_bytree', 0.8),
                'random_state': self.random_state,
                'n_jobs': -1,
                'verbose': False
            }

            # Add GPU support if requested
            # XGBoost 3.1+ uses device='cuda:0', 'cuda:1', etc.
            if self.device.startswith('cuda'):
                # Use device parameter directly (XGBoost 3.1+)
                # For 'cuda' without ID, default to 'cuda:0'
                if self.device == 'cuda':
                    xgb_params['device'] = 'cuda:0'
                else:
                    # Use specific device like 'cuda:1'
                    xgb_params['device'] = self.device
                xgb_params['tree_method'] = 'hist'
            # For CPU, no need to set device parameter (default is CPU)

            return xgb.XGBRegressor(**xgb_params)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def fit(self, X, y):
        self.feature_dim = X.shape[1]
        self.model = self._create_model(self.feature_dim)
        self.model.fit(X, y)
        return self

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model not fitted yet")
        return self.model.predict(X)

    def score(self, X, y):
        return self.model.score(X, y)


def run_variant_experiment(
    train_file,
    test_file,
    variant,
    model_type='ridge',
    device='cuda',
    random_state=42,
    llm_field=None,
    llm_fields=None,
    llm_input_mode=None,
    llm_input_name=None,
    sample_per_llm=10,
    llm_sampling_strategy="all",
    llm_vector_transform="raw",
    llm_dim=None,
    result_file_name=None,
    save_split_results=True,
    result_precision=4,
    **model_kwargs
):
    """
    Run variant experiment

    Args:
        train_file: Training data file path (JSON or parquet)
        test_file: Test data file path (JSON or parquet)
        variant: Feature variant ('x_only', 'x_avg_llm', 'x_all_llm',
                 'x_one_llm', 'one_logprob')
        model_type: Model type ('ridge', 'rf', 'mlp', 'xgboost')
        device: Device for training ('cuda' for GPU, 'cpu' for CPU)
        random_state: Random seed for stochastic models
        llm_field: Optional custom LLM response field (e.g., 'claude-3.5-haiku_norm')
        llm_fields: Optional list/comma-separated LLM fields to concatenate in order
        llm_input_mode: Optional label describing the LLM input construction
        llm_input_name: Optional human-readable name for the LLM input construction
        sample_per_llm: Responses sampled per field when using sample_per_field
        llm_sampling_strategy: "all" or "sample_per_field"
        llm_vector_transform: Canonicalization for vector-valued LLM inputs
        llm_dim: Optional LLM response dimension for x_all_llm ablation.
                 If None, uses max available length across train+test.
        result_file_name: Optional result filename under results/
        save_split_results: Whether to also save results/train and results/test CSV rows
        result_precision: Number of decimal places retained in result CSV files
        **model_kwargs: Model parameters
    """
    print("=" * 60)
    print(f"Running variant experiment: {variant} + {model_type}")
    print("=" * 60)

    # Load data
    train_data = load_data(train_file)
    test_data = load_data(test_file)

    # Detect data format
    is_individual = is_individual_data(train_data)
    data_format = "Individual" if is_individual else "Average"
    print(f"Data format: {data_format}")

    print(f"Train samples: {len(train_data)}, Test samples: {len(test_data)}")

    normalized_llm_fields = _normalize_llm_fields(llm_fields)
    if llm_field and normalized_llm_fields:
        raise ValueError("Use either --llm_field or --llm_fields, not both.")
    if variant == ONE_LOGPROB_VARIANT:
        if normalized_llm_fields:
            raise ValueError("one_logprob accepts one probability field, not --llm_fields.")
        if llm_sampling_strategy != "all":
            raise ValueError("one_logprob does not support LLM response sampling.")
        if llm_vector_transform != "raw":
            raise ValueError("one_logprob probabilities must use the raw vector order.")
        if llm_dim is not None:
            raise ValueError("one_logprob uses the materialized fixed width; omit --llm_dim.")
        if llm_field is None:
            llm_field = ONE_LOGPROB_FIELD

    # Strict validation for custom llm field: no alias fallback, and key must exist.
    if llm_field:
        missing_train = sum(1 for item in train_data if llm_field not in item)
        missing_test = sum(1 for item in test_data if llm_field not in item)
        if missing_train > 0 or missing_test > 0:
            raise ValueError(
                f"--llm_field '{llm_field}' not found in all rows "
                f"(train missing: {missing_train}, test missing: {missing_test})."
            )
    if normalized_llm_fields:
        missing_train = sum(
            1 for item in train_data if any(field not in item for field in normalized_llm_fields)
        )
        missing_test = sum(
            1 for item in test_data if any(field not in item for field in normalized_llm_fields)
        )
        if missing_train > 0 or missing_test > 0:
            raise ValueError(
                f"--llm_fields contains fields not found in all rows "
                f"(train missing rows: {missing_train}, test missing rows: {missing_test})."
            )

    # Prepare features
    print(f"Preparing features (variant={variant})...")
    if llm_field:
        print(f"  Using LLM field: {llm_field}")
    if normalized_llm_fields:
        print(f"  Using LLM fields ({len(normalized_llm_fields)}): {', '.join(normalized_llm_fields)}")
    print(f"  Random state: {random_state}")
    if llm_sampling_strategy == "sample_per_field":
        print(f"  Sample per LLM: {sample_per_llm}")
    print(f"  LLM sampling strategy: {llm_sampling_strategy}")
    print(f"  LLM vector transform: {llm_vector_transform}")
    if llm_dim is not None and variant not in {'x_all_llm', 'x_avg_llm'}:
        print(f"  Note: --llm_dim is only used by x_all_llm / x_avg_llm, ignored for variant={variant}.")
    effective_llm_dim = llm_dim
    output_llm_responses_length = None
    if variant == 'x_all_llm':
        max_llm_len = get_max_llm_response_length(
            train_data + test_data,
            llm_field=llm_field,
            llm_fields=normalized_llm_fields,
            llm_sampling_strategy=llm_sampling_strategy,
            sample_per_llm=sample_per_llm,
            random_state=random_state,
        )
        if max_llm_len <= 0:
            raise ValueError(
                "x_all_llm requires non-empty LLM responses, but max available length is 0."
            )
        if llm_dim is None:
            effective_llm_dim = max_llm_len
        else:
            if llm_dim <= 0:
                raise ValueError(f"--llm_dim must be positive, got {llm_dim}.")
            if llm_dim > max_llm_len:
                raise ValueError(
                    f"--llm_dim ({llm_dim}) is greater than max available LLM length "
                    f"({max_llm_len})."
                )
        output_llm_responses_length = int(effective_llm_dim)
        print(f"  LLM dim for x_all_llm: {effective_llm_dim} (max available: {max_llm_len})")
    elif variant == ONE_LOGPROB_VARIANT:
        vector_lengths = {
            len(_to_llm_response_list(item.get(llm_field)))
            for item in train_data + test_data
        }
        if len(vector_lengths) != 1 or next(iter(vector_lengths)) <= 0:
            raise ValueError(
                "one_logprob requires one consistent, non-empty vector width "
                "across train and test"
            )
        output_llm_responses_length = int(next(iter(vector_lengths)))
        print(f"  One Logprob vector dim: {output_llm_responses_length}")
    X_train, y_train, orig_train, ranges_train, baseline_train, twin_ids_train = prepare_features(
        train_data,
        variant,
        llm_field,
        normalized_llm_fields,
        sample_per_llm,
        effective_llm_dim,
        llm_sampling_strategy,
        random_state,
        llm_vector_transform,
    )
    X_test, y_test, orig_test, ranges_test, baseline_test, twin_ids_test = prepare_features(
        test_data,
        variant,
        llm_field,
        normalized_llm_fields,
        sample_per_llm,
        effective_llm_dim,
        llm_sampling_strategy,
        random_state,
        llm_vector_transform,
    )

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
        print("ERROR: No valid training or test data!")
        print("   Please ensure dataset has required embedding fields")
        return None

    # Train model
    print(f"Training model ({model_type})...")
    if device.startswith('cuda'):
        print(f"Device: {device}")
    else:
        print(f"Device: {device}")
    model = VariantDebiasModel(
        model_type,
        device=device,
        random_state=random_state,
        **model_kwargs,
    )
    model.fit(X_train, y_train)

    # Predict (normalized space)
    y_pred_train_norm = model.predict(X_train)
    y_pred_test_norm = model.predict(X_test)

    # Map normalized predictions back to original space
    y_pred_train_original = np.array([
        y_pred_train_norm[i] * (ranges_train[i][1] - ranges_train[i][0]) + ranges_train[i][0]
        for i in range(len(y_pred_train_norm))
    ])

    y_pred_test_original = np.array([
        y_pred_test_norm[i] * (ranges_test[i][1] - ranges_test[i][0]) + ranges_test[i][0]
        for i in range(len(y_pred_test_norm))
    ])

    # Map baseline back to original space
    baseline_test_original = np.array([
        baseline_test[i] * (ranges_test[i][1] - ranges_test[i][0]) + ranges_test[i][0]
        for i in range(len(baseline_test))
    ])

    # Compute baseline train set original space values
    baseline_train_original = np.array([
        baseline_train[i] * (ranges_train[i][1] - ranges_train[i][0]) + ranges_train[i][0]
        for i in range(len(baseline_train))
    ])

    # Compute metrics (normalized space without improvement)
    train_metrics_norm = compute_metrics(y_train, y_pred_train_norm, baseline_train, compute_improvement=False)
    test_metrics_norm = compute_metrics(y_test, y_pred_test_norm, baseline_test, compute_improvement=False)
    train_metrics_original = compute_metrics(orig_train, y_pred_train_original, compute_improvement=False)
    test_metrics_original = compute_metrics(orig_test, y_pred_test_original, baseline_test_original, compute_improvement=True)

    # Compute accuracy metrics (original space)
    train_acc_baseline_hard = compute_accuracy_hard(orig_train, baseline_train_original, ranges_train)
    train_acc_baseline_soft = compute_accuracy_soft(orig_train, baseline_train_original, ranges_train)
    train_acc_baseline_mad = compute_accuracy_mad(orig_train, baseline_train_original, ranges_train)
    train_acc_model_hard = compute_accuracy_hard(orig_train, y_pred_train_original, ranges_train)
    train_acc_model_soft = compute_accuracy_soft(orig_train, y_pred_train_original, ranges_train)
    train_acc_model_mad = compute_accuracy_mad(orig_train, y_pred_train_original, ranges_train)

    test_acc_baseline_hard = compute_accuracy_hard(orig_test, baseline_test_original, ranges_test)
    test_acc_baseline_soft = compute_accuracy_soft(orig_test, baseline_test_original, ranges_test)
    test_acc_baseline_mad = compute_accuracy_mad(orig_test, baseline_test_original, ranges_test)
    test_acc_model_hard = compute_accuracy_hard(orig_test, y_pred_test_original, ranges_test)
    test_acc_model_soft = compute_accuracy_soft(orig_test, y_pred_test_original, ranges_test)
    test_acc_model_mad = compute_accuracy_mad(orig_test, y_pred_test_original, ranges_test)

    # Add accuracy to metrics dict
    train_metrics_original['acc_baseline_hard'] = train_acc_baseline_hard
    train_metrics_original['acc_baseline_soft'] = train_acc_baseline_soft
    train_metrics_original['acc_baseline_mad'] = train_acc_baseline_mad
    train_metrics_original['acc_model_hard'] = train_acc_model_hard
    train_metrics_original['acc_model_soft'] = train_acc_model_soft
    train_metrics_original['acc_model_mad'] = train_acc_model_mad

    test_metrics_original['acc_baseline_hard'] = test_acc_baseline_hard
    test_metrics_original['acc_baseline_soft'] = test_acc_baseline_soft
    test_metrics_original['acc_baseline_mad'] = test_acc_baseline_mad
    test_metrics_original['acc_model_hard'] = test_acc_model_hard
    test_metrics_original['acc_model_soft'] = test_acc_model_soft
    test_metrics_original['acc_model_mad'] = test_acc_model_mad

    # Extended metrics: correlation, variance ratio, Glass's delta,
    # per-individual correlation, individuation gap, range coverage ratio.
    from debias.evaluate_variants import compute_extended_metrics
    ext_model = compute_extended_metrics(orig_test, y_pred_test_original, twin_ids_test)
    ext_base = compute_extended_metrics(orig_test, baseline_test_original, twin_ids_test)
    for k, v in ext_model.items():
        test_metrics_original[f'model_{k}'] = v
    for k, v in ext_base.items():
        test_metrics_original[f'baseline_{k}'] = v

    # Print results
    print_experiment_results(variant, model_type, train_metrics_norm, test_metrics_norm, test_metrics_original, train_metrics_original)

    # Save to CSV (infer result file name from training file path)
    save_single_result_to_csv(
        variant,
        model_type,
        train_metrics_norm,
        test_metrics_norm,
        test_metrics_original,
        train_file,
        train_metrics_original,
        llm_field=llm_field,
        llm_fields=normalized_llm_fields,
        llm_input_mode=llm_input_mode,
        llm_input_name=llm_input_name,
        llm_responses_length=output_llm_responses_length,
        llm_vector_transform=llm_vector_transform,
        result_file_name=result_file_name,
        save_split_results=save_split_results,
        result_precision=result_precision,
    )

    return {
        'variant': variant,
        'model_type': model_type,
        'train_metrics_norm': train_metrics_norm,
        'test_metrics_norm': test_metrics_norm,
        'train_metrics_original': train_metrics_original,
        'test_metrics_original': test_metrics_original,
        'model': model
    }


def run_all_variants(
    train_file,
    test_file,
    model_type='ridge',
    device='cuda',
    random_state=42,
    llm_field=None,
    llm_fields=None,
    llm_input_mode=None,
    llm_input_name=None,
    sample_per_llm=10,
    llm_sampling_strategy="all",
    llm_vector_transform="raw",
    llm_dim=None,
    result_file_name=None,
    save_split_results=True,
    result_precision=4,
    **model_kwargs
):
    """Run all variant experiments"""
    variants = [
        'x_only',
        'x_avg_llm',
        'x_all_llm',
        'x_one_llm'
    ]

    results = {}

    for variant in variants:
        try:
            result = run_variant_experiment(
                train_file,
                test_file,
                variant,
                model_type,
                device=device,
                random_state=random_state,
                llm_field=llm_field,
                llm_fields=llm_fields,
                llm_input_mode=llm_input_mode,
                llm_input_name=llm_input_name,
                sample_per_llm=sample_per_llm,
                llm_sampling_strategy=llm_sampling_strategy,
                llm_vector_transform=llm_vector_transform,
                llm_dim=llm_dim,
                result_file_name=result_file_name,
                save_split_results=save_split_results,
                result_precision=result_precision,
                **model_kwargs
            )
            results[variant] = result
        except Exception as e:
            print(f"ERROR in {variant}: {e}")
            import traceback
            traceback.print_exc()
            results[variant] = {'error': str(e)}

    # Print summary
    print("\n" + "=" * 80)
    print("All variants summary")
    print("=" * 80)
    print(f"{'Variant':<20} {'Test MSE':<12} {'Test MAE':<12} {'R²':<10}")
    print("-" * 80)

    for variant in variants:
        if variant in results and 'test_metrics_norm' in results[variant]:
            m = results[variant]['test_metrics_norm']
            print(f"{variant:<20} {m['mse']:<12.6f} {m['mae']:<12.6f} {m['r2']:<10.4f}")
        elif variant in results and 'error' in results[variant]:
            print(f"{variant:<20} ERROR: {results[variant]['error'][:50]}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run debias experiments with different feature combinations"
    )
    parser.add_argument(
        "--train_file",
        default="dataset/EEDI/Aggreated/eedi_train.parquet",
        help="Training data file path (parquet)"
    )
    parser.add_argument(
        "--test_file",
        default="dataset/EEDI/Aggreated/eedi_test.parquet",
        help="Test data file path (parquet)"
    )
    parser.add_argument(
        "--variant",
        choices=[
            'x_only',
            'x_avg_llm',
            'x_all_llm',
            'x_one_llm',
            ONE_LOGPROB_VARIANT,
            'all',
        ],
        default='all',
        help="Feature variant (use 'all' to run all variants)"
    )
    parser.add_argument(
        "--model_type",
        choices=['ridge', 'rf', 'mlp', 'xgboost'],
        default='ridge',
        help="Model type"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Ridge regularization coefficient"
    )
    parser.add_argument(
        "--n_estimators",
        type=int,
        default=100,
        help="RF/XGBoost number of trees"
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        default=10,
        help="RF/XGBoost max depth"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.05,
        help="XGBoost learning rate (default: 0.05)"
    )
    parser.add_argument(
        "--subsample",
        type=float,
        default=0.8,
        help="XGBoost subsample ratio (default: 0.8)"
    )
    parser.add_argument(
        "--colsample_bytree",
        type=float,
        default=0.8,
        help="XGBoost column sample by tree ratio (default: 0.8)"
    )
    parser.add_argument(
        "--hidden_layers",
        type=str,
        default="2048,1024,512,256",
        help="MLP hidden layer sizes (comma-separated), e.g., '2048,1024,512,256'"
    )
    parser.add_argument(
        "--mlp_alpha",
        type=float,
        default=0.00001,
        help="MLP L2 regularization parameter"
    )
    parser.add_argument(
        "--learning_rate_init",
        type=float,
        default=0.0005,
        help="MLP initial learning rate"
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=3000,
        help="MLP max iterations"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="MLP batch size"
    )
    parser.add_argument(
        "--mlp_dropout",
        type=float,
        default=0.0,
        help="MLP dropout rate for PyTorch MLP (default: 0.0)"
    )
    parser.add_argument(
        "--validation_fraction",
        type=float,
        default=0.1,
        help="Validation split fraction used by MLP early stopping (default: 0.1)"
    )
    parser.add_argument(
        "--n_iter_no_change",
        type=int,
        default=30,
        help="MLP early stopping patience in epochs (default: 30)"
    )
    parser.add_argument(
        "--min_delta",
        type=float,
        default=1e-6,
        help="Minimum validation loss improvement to reset patience for MLP (default: 1e-6)"
    )
    parser.add_argument(
        "--no_mlp_standardize",
        action="store_true",
        help="Disable feature standardization for MLP"
    )
    parser.add_argument(
        "--llm_field",
        type=str,
        default=None,
        help=(
            "Custom LLM response field to use (e.g., 'LLM_Responses_norm'). "
            "If set, field name must exist exactly (no alias fallback). "
            "If not specified, uses 'LLM_Responses_norm'."
        )
    )
    parser.add_argument(
        "--llm_fields",
        type=str,
        default=None,
        help=(
            "Comma-separated custom LLM response fields to concatenate in order "
            "(e.g., 'gpt-4o_norm,deepseek-v3_norm'). "
            "Used for multi-source x_all_llm / x_avg_llm ablations."
        )
    )
    parser.add_argument(
        "--llm_input_mode",
        type=str,
        default=None,
        help="Optional metadata label for the LLM input construction (e.g., single_source, sample_multisource, concat_multisource)."
    )
    parser.add_argument(
        "--llm_input_name",
        type=str,
        default=None,
        help="Optional human-readable metadata label for the LLM input construction."
    )
    parser.add_argument(
        "--sample_per_llm",
        type=int,
        default=10,
        help=(
            "Number of responses sampled per LLM field when "
            "--llm_sampling_strategy=sample_per_field (default: 10)"
        )
    )
    parser.add_argument(
        "--llm_sampling_strategy",
        type=str,
        default="all",
        choices=["all", "sample_per_field"],
        help="How to construct the LLM input vector from llm_field/llm_fields."
    )
    parser.add_argument(
        "--llm_vector_transform",
        type=str,
        default="raw",
        choices=VALID_LLM_VECTOR_TRANSFORMS,
        help=(
            "Canonicalization for vector-valued LLM inputs. "
            "'raw' keeps the original order, 'sorted' sorts responses within each field, "
            "and 'centered_sorted' sorts mean-centered responses within each field."
        )
    )
    parser.add_argument(
        "--llm_dim",
        type=int,
        default=None,
        help=(
            "Optional LLM response dimension for x_all_llm ablation. "
            "Default: use max available length across train+test. "
            "If set greater than max length, raises an error."
        )
    )
    parser.add_argument(
        "--device",
        type=str,
        default='cuda',
        choices=['cpu', 'cuda'] + [f'cuda:{idx}' for idx in range(8)],
        help="Device for training (default: cuda). For multi-GPU systems, use 'cuda:0', 'cuda:1', etc."
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for stochastic models (default: 42)"
    )
    parser.add_argument(
        "--result_file",
        type=str,
        default=None,
        help=(
            "Optional output CSV path under results/ "
            "(e.g., aggreated_twin_llm_dim_ablation_result.csv or group/twin/seed42.csv)."
        )
    )
    parser.add_argument(
        "--no_split_results",
        action="store_true",
        help="Only write the combined results CSV under results/, skipping results/train and results/test."
    )
    parser.add_argument(
        "--result_precision",
        type=int,
        default=4,
        help="Decimal places retained in result CSV metrics (default: 4)."
    )

    args = parser.parse_args()

    if args.llm_field:
        print(f"Using custom LLM field: {args.llm_field}")
    if args.llm_fields:
        print(f"Using custom LLM fields: {args.llm_fields}")
    if args.llm_field and args.llm_fields:
        raise ValueError("Use either --llm_field or --llm_fields, not both.")

    if args.variant == 'all':
        run_all_variants(
            args.train_file,
            args.test_file,
            args.model_type,
            device=args.device,
            random_state=args.random_state,
            llm_field=args.llm_field,
            llm_fields=args.llm_fields,
            llm_input_mode=args.llm_input_mode,
            llm_input_name=args.llm_input_name,
            sample_per_llm=args.sample_per_llm,
            llm_sampling_strategy=args.llm_sampling_strategy,
            llm_vector_transform=args.llm_vector_transform,
            llm_dim=args.llm_dim,
            result_file_name=args.result_file,
            save_split_results=not args.no_split_results,
            result_precision=args.result_precision,
            alpha=args.alpha,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            subsample=args.subsample,
            colsample_bytree=args.colsample_bytree,
            hidden_layers=args.hidden_layers,
            mlp_alpha=args.mlp_alpha,
            learning_rate_init=args.learning_rate_init,
            max_iter=args.max_iter,
            batch_size=args.batch_size,
            mlp_dropout=args.mlp_dropout,
            validation_fraction=args.validation_fraction,
            n_iter_no_change=args.n_iter_no_change,
            min_delta=args.min_delta,
            mlp_standardize=not args.no_mlp_standardize
        )
    else:
        run_variant_experiment(
            args.train_file,
            args.test_file,
            args.variant,
            args.model_type,
            device=args.device,
            random_state=args.random_state,
            llm_field=args.llm_field,
            llm_fields=args.llm_fields,
            llm_input_mode=args.llm_input_mode,
            llm_input_name=args.llm_input_name,
            sample_per_llm=args.sample_per_llm,
            llm_sampling_strategy=args.llm_sampling_strategy,
            llm_vector_transform=args.llm_vector_transform,
            llm_dim=args.llm_dim,
            result_file_name=args.result_file,
            save_split_results=not args.no_split_results,
            result_precision=args.result_precision,
            alpha=args.alpha,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            subsample=args.subsample,
            colsample_bytree=args.colsample_bytree,
            hidden_layers=args.hidden_layers,
            mlp_alpha=args.mlp_alpha,
            learning_rate_init=args.learning_rate_init,
            max_iter=args.max_iter,
            batch_size=args.batch_size,
            mlp_dropout=args.mlp_dropout,
            validation_fraction=args.validation_fraction,
            n_iter_no_change=args.n_iter_no_change,
            min_delta=args.min_delta,
            mlp_standardize=not args.no_mlp_standardize
        )
