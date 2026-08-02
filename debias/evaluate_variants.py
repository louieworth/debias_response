"""
评估指标计算模块

包含：
- Accuracy 计算（硬分类、软分类和MAD）
- 基础回归指标（MSE, MAE, R²等）
- 结果打印和保存
"""

import csv
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pathlib import Path


def _normalize_score_bounds(min_val, max_val):
    """
    Convert score range bounds to integer category bounds.

    Parquet/object loading may produce float bounds such as 1.0/5.0.
    Accuracy metrics use discrete options and therefore need int bounds.
    """
    min_f = float(min_val)
    max_f = float(max_val)
    if min_f > max_f:
        min_f, max_f = max_f, min_f

    if np.isclose(min_f, round(min_f)):
        min_i = int(round(min_f))
    else:
        min_i = int(np.floor(min_f))

    if np.isclose(max_f, round(max_f)):
        max_i = int(round(max_f))
    else:
        max_i = int(np.ceil(max_f))

    if min_i > max_i:
        min_i, max_i = max_i, min_i
    return min_i, max_i


def get_two_nearest_weights(value, options):
    """
    根据值计算最近两个选项的权重

    Args:
        value: 连续值（如 1.5, 2.3）
        options: 离散选项列表（如 [1,2,3,4,5]）

    Returns:
        字典 {选项: 权重}

    Examples:
        value=1.5, options=[1,2,3,4,5]
        → {1: 0.5, 2: 0.5}

        value=2.3, options=[1,2,3,4,5]
        → {2: 0.7, 3: 0.3}

        value=2.0, options=[1,2,3,4,5]（正好是某个选项）
        → {2: 1.0}
    """
    # 计算到所有选项的距离
    distances = [(opt, abs(opt - value)) for opt in options]

    # 找最近的两个选项
    distances_sorted = sorted(distances, key=lambda x: x[1])
    nearest_two = distances_sorted[:2]

    # 如果正好在某个选项上，只返回那个选项
    if nearest_two[0][1] < 1e-6:  # 距离接近0
        return {nearest_two[0][0]: 1.0}

    # 否则，根据距离分配权重（距离越近权重越大）
    opt1, dist1 = nearest_two[0]
    opt2, dist2 = nearest_two[1]

    # 权重与距离成反比
    total_dist = dist1 + dist2
    weight1 = dist2 / total_dist  # 距离越小，权重越大
    weight2 = dist1 / total_dist

    return {opt1: weight1, opt2: weight2}


def compute_accuracy_hard(y_true, y_pred, score_ranges):
    """
    四舍五入后的严格匹配准确率

    Args:
        y_true: 真实值数组（原始空间）
        y_pred: 预测值数组（原始空间）
        score_ranges: 每个样本的分数范围

    Returns:
        准确率（0-1之间）

    Examples:
        y_true=1.5, y_pred=1.5 → both → 2 → match → 1.0
        y_true=2.0, y_pred=2.4 → 2 vs 2 → match → 1.0
        y_true=2.0, y_pred=2.6 → 2 vs 3 → no match → 0.0
    """
    correct = 0
    for i in range(len(y_true)):
        min_val, max_val = _normalize_score_bounds(*score_ranges[i])

        # 四舍五入到离散选项
        true_rounded = int(round(y_true[i]))
        pred_rounded = int(round(y_pred[i]))

        # 边界处理
        true_rounded = max(min_val, min(max_val, true_rounded))
        pred_rounded = max(min_val, min(max_val, pred_rounded))

        if true_rounded == pred_rounded:
            correct += 1

    return correct / len(y_true)


def compute_accuracy_soft(y_true, y_pred, score_ranges):
    """
    软准确率：只考虑最近的两个选项，根据距离分配权重

    Args:
        y_true: 真实值数组（原始空间）
        y_pred: 预测值数组（原始空间）
        score_ranges: 每个样本的分数范围

    Returns:
        准确率（0-1之间）

    Examples:
        y_true=2, y_pred=1.5
        - weights_true = {2: 1.0}
        - weights_pred = {1: 0.5, 2: 0.5}
        - accuracy = 1.0 * 0.5 = 0.5

        y_true=1.5, y_pred=2.3
        - weights_true = {1: 0.5, 2: 0.5}
        - weights_pred = {2: 0.7, 3: 0.3}
        - accuracy = 0.5*0.7 = 0.35
    """
    total_score = 0.0

    for i in range(len(y_true)):
        min_val, max_val = _normalize_score_bounds(*score_ranges[i])
        options = list(range(min_val, max_val + 1))

        # 计算 y_true 的权重（最近两个选项）
        weights_true = get_two_nearest_weights(y_true[i], options)

        # 计算 y_pred 的权重（最近两个选项）
        weights_pred = get_two_nearest_weights(y_pred[i], options)

        # 计算重叠度（两个权重分布的交集）
        score = 0.0
        for opt in options:
            if opt in weights_true and opt in weights_pred:
                score += weights_true[opt] * weights_pred[opt]

        total_score += score

    return total_score / len(y_true)


def compute_accuracy_mad(y_true, y_pred, score_ranges):
    """
    MAD-based accuracy: mean(1 - |y_pred - y_true| / (score_max - score_min))

    Args:
        y_true: 真实值数组（原始空间）
        y_pred: 预测值数组（原始空间）
        score_ranges: 每个样本的分数范围

    Returns:
        平均 MAD-based accuracy（通常在 0-1，越界预测时可 <0）
    """
    total_score = 0.0

    for i in range(len(y_true)):
        min_val = float(score_ranges[i][0])
        max_val = float(score_ranges[i][1])
        if min_val > max_val:
            min_val, max_val = max_val, min_val

        range_width = max_val - min_val
        if np.isclose(range_width, 0.0):
            score = 1.0 if np.isclose(float(y_true[i]), float(y_pred[i])) else 0.0
        else:
            score = 1.0 - abs(float(y_pred[i]) - float(y_true[i])) / range_width

        total_score += score

    return total_score / len(y_true)


def _pearson_r(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) < 2:
        return np.nan
    s1 = y_true.std()
    s2 = y_pred.std()
    if s1 < 1e-12 or s2 < 1e-12:
        return np.nan
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def _spearman_r(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) < 2:
        return np.nan
    # rankdata without scipy dependency
    def _rank(a):
        order = a.argsort()
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(len(a), dtype=float)
        # average ties
        _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
        sums = np.zeros(len(counts))
        np.add.at(sums, inv, ranks)
        avg = sums / counts
        return avg[inv]
    return _pearson_r(_rank(y_true), _rank(y_pred))


def _fisher_z(r):
    if r is None or np.isnan(r):
        return np.nan
    r = float(np.clip(r, -0.999999, 0.999999))
    return 0.5 * np.log((1 + r) / (1 - r))


def compute_correlation_metrics(y_true, y_pred):
    """
    Pearson r, Spearman rho, Fisher-z of Pearson r (across all samples,
    i.e. individual-level heterogeneity correlation).
    """
    r = _pearson_r(y_true, y_pred)
    rho = _spearman_r(y_true, y_pred)
    return {
        "pearson_r": r,
        "spearman_r": rho,
        "fisher_z": _fisher_z(r),
    }


def compute_variance_ratio(y_true, y_pred):
    """std(pred) / std(true). Ideal value 1.0; <1 indicates homogenization."""
    s_true = float(np.std(np.asarray(y_true, dtype=float)))
    s_pred = float(np.std(np.asarray(y_pred, dtype=float)))
    if s_true < 1e-12:
        return np.nan
    return s_pred / s_true


def compute_glass_delta(y_true, y_pred):
    """(mean(pred) - mean(true)) / std(true). Signed aggregate bias; 0 ideal."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    s_true = float(np.std(y_true))
    if s_true < 1e-12:
        return np.nan
    return float((y_pred.mean() - y_true.mean()) / s_true)


def _group_indices(ids):
    """Return dict id -> list of indices. None ids are dropped."""
    groups = {}
    if ids is None:
        return groups
    for i, g in enumerate(ids):
        if g is None:
            continue
        groups.setdefault(g, []).append(i)
    return groups


def compute_per_individual_correlations(y_true, y_pred, group_ids, min_group_size=3):
    """
    For each person (group_ids value), compute Pearson and Spearman between
    their items' y_true and y_pred; return the mean over persons with at least
    min_group_size answered questions.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    groups = _group_indices(group_ids)
    pearsons, spearmans = [], []
    for idx in groups.values():
        if len(idx) < min_group_size:
            continue
        r = _pearson_r(y_true[idx], y_pred[idx])
        rho = _spearman_r(y_true[idx], y_pred[idx])
        if not np.isnan(r):
            pearsons.append(r)
        if not np.isnan(rho):
            spearmans.append(rho)
    return {
        "per_person_pearson": float(np.mean(pearsons)) if pearsons else np.nan,
        "per_person_spearman": float(np.mean(spearmans)) if spearmans else np.nan,
        "per_person_groups_used": len(pearsons),
    }


def compute_individuation_gap(y_true, y_pred, group_ids, min_group_size=3):
    """
    Pearson(across-all) minus mean Pearson within persons.
    Positive value => model mostly captures between-person variation;
    small/negative => model also resolves within-person structure.
    """
    across = _pearson_r(y_true, y_pred)
    within = compute_per_individual_correlations(
        y_true, y_pred, group_ids, min_group_size=min_group_size
    )["per_person_pearson"]
    if np.isnan(across) or np.isnan(within):
        return np.nan
    return float(across - within)


def compute_range_coverage_ratio(y_true, y_pred, group_ids, min_group_size=3):
    """
    Mean over persons of (max_pred - min_pred) / (max_true - min_true).
    1.0 means predicted dynamic range matches true; <1 means flattening.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    groups = _group_indices(group_ids)
    ratios = []
    for idx in groups.values():
        if len(idx) < min_group_size:
            continue
        rng_true = float(y_true[idx].max() - y_true[idx].min())
        rng_pred = float(y_pred[idx].max() - y_pred[idx].min())
        if rng_true < 1e-12:
            continue
        ratios.append(rng_pred / rng_true)
    if not ratios:
        return np.nan
    return float(np.mean(ratios))


def compute_extended_metrics(y_true, y_pred, group_ids=None):
    """
    Bundle the new reporting metrics so they can be attached to *_metrics_original.
    All values are scalars (None/NaN if not applicable).
    """
    out = {}
    out.update(compute_correlation_metrics(y_true, y_pred))
    out["variance_ratio"] = compute_variance_ratio(y_true, y_pred)
    out["glass_delta"] = compute_glass_delta(y_true, y_pred)
    if group_ids is not None and len(group_ids) == len(y_true):
        per = compute_per_individual_correlations(y_true, y_pred, group_ids)
        out.update(per)
        out["individuation_gap"] = compute_individuation_gap(y_true, y_pred, group_ids)
        out["range_coverage_ratio"] = compute_range_coverage_ratio(y_true, y_pred, group_ids)
    else:
        out["per_person_pearson"] = np.nan
        out["per_person_spearman"] = np.nan
        out["per_person_groups_used"] = 0
        out["individuation_gap"] = np.nan
        out["range_coverage_ratio"] = np.nan
    return out


def compute_metrics(y_true, y_pred, y_baseline_llm=None, compute_improvement=True):
    """
    计算评估指标

    Args:
        y_true: 真实值
        y_pred: 预测值
        y_baseline_llm: 基线LLM响应（可选）
        compute_improvement: 是否计算improvement百分比（只在原始空间计算）

    Returns:
        包含各种指标的字典
    """
    metrics = {
        'mse': mean_squared_error(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'r2': r2_score(y_true, y_pred),
        'avg_human_response': np.mean(y_true),
        'avg_debiased_response': np.mean(y_pred),
    }

    # 计算基线指标
    if y_baseline_llm is not None:
        metrics['mse_baseline'] = mean_squared_error(y_true, y_baseline_llm)
        metrics['mae_baseline'] = mean_absolute_error(y_true, y_baseline_llm)

    # 只在原始空间计算improvement百分比
    if compute_improvement and 'mse_baseline' in metrics:
        if metrics['mse_baseline'] > 0:
            metrics['mse_improvement_pct'] = (1 - metrics['mse'] / metrics['mse_baseline']) * 100
        else:
            metrics['mse_improvement_pct'] = 0
        if metrics['mae_baseline'] > 0:
            metrics['mae_improvement_pct'] = (1 - metrics['mae'] / metrics['mae_baseline']) * 100
        else:
            metrics['mae_improvement_pct'] = 0

    return metrics


def print_experiment_results(variant, model_type, train_metrics, test_metrics, test_metrics_original, train_metrics_original=None):
    """
    打印实验结果

    Args:
        variant: 特征变体名称
        model_type: 模型类型
        train_metrics: 训练集指标（归一化空间）
        test_metrics: 测试集指标（归一化空间）
        test_metrics_original: 测试集指标（原始空间）
        train_metrics_original: 训练集指标（原始空间，可选）
    """
    print("\n" + "=" * 60)
    print(f"结果: {variant} + {model_type}")
    print("=" * 60)

    print("\n归一化空间指标:")
    print(f"  训练集 - MSE: {train_metrics['mse']:.6f}, MAE: {train_metrics['mae']:.6f}, R²: {train_metrics['r2']:.4f}")
    print(f"  测试集 - MSE: {test_metrics['mse']:.6f}, MAE: {test_metrics['mae']:.6f}, R²: {test_metrics['r2']:.4f}")

    if 'mse_improvement_pct' in test_metrics:
        print(f"  vs LLM基线 - MSE改进: {test_metrics['mse_improvement_pct']:.2f}%, MAE改进: {test_metrics['mae_improvement_pct']:.2f}%")

    print("\n原始空间指标:")
    print(f"  测试集 - MSE: {test_metrics_original['mse']:.6f}, MAE: {test_metrics_original['mae']:.6f}, R²: {test_metrics_original['r2']:.4f}")

    # 打印 accuracy 指标
    if train_metrics_original is not None and 'acc_baseline_hard' in train_metrics_original:
        print(f"\n准确率指标:")
        print(f"  训练集 - Base Hard: {train_metrics_original['acc_baseline_hard']:.4f}, Model Hard: {train_metrics_original['acc_model_hard']:.4f}")
        print(f"  训练集 - Base Soft: {train_metrics_original['acc_baseline_soft']:.4f}, Model Soft: {train_metrics_original['acc_model_soft']:.4f}")
        if 'acc_baseline_mad' in train_metrics_original:
            print(f"  训练集 - Base MAD : {train_metrics_original['acc_baseline_mad']:.4f}, Model MAD : {train_metrics_original['acc_model_mad']:.4f}")
        print(f"  测试集 - Base Hard: {test_metrics_original['acc_baseline_hard']:.4f}, Model Hard: {test_metrics_original['acc_model_hard']:.4f}")
        print(f"  测试集 - Base Soft: {test_metrics_original['acc_baseline_soft']:.4f}, Model Soft: {test_metrics_original['acc_model_soft']:.4f}")
        if 'acc_baseline_mad' in test_metrics_original:
            print(f"  测试集 - Base MAD : {test_metrics_original['acc_baseline_mad']:.4f}, Model MAD : {test_metrics_original['acc_model_mad']:.4f}")


def _safe_float(value):
    if value is None:
        return np.nan
    if isinstance(value, str) and value.strip() == "":
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _improvement_pct(base_value, model_value, higher_is_better=False):
    base_f = _safe_float(base_value)
    model_f = _safe_float(model_value)
    if not np.isfinite(base_f) or not np.isfinite(model_f):
        return np.nan
    if np.isclose(base_f, 0.0):
        return 0.0
    if higher_is_better:
        return (model_f - base_f) / abs(base_f) * 100.0
    return (base_f - model_f) / abs(base_f) * 100.0


def _format_metric_value(value, precision=4):
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "":
            return ""
        numeric = _safe_float(stripped)
        if np.isfinite(numeric):
            return f"{numeric:.{precision}f}"
        return stripped

    if isinstance(value, (int, np.integer)):
        return str(int(value))

    if isinstance(value, (float, np.floating)):
        numeric = _safe_float(value)
        if np.isfinite(numeric):
            return f"{numeric:.{precision}f}"
        return ""

    if value is None:
        return ""

    return str(value)


def _append_row_to_csv(output_file: Path, row: dict):
    import fcntl
    output_file.parent.mkdir(parents=True, exist_ok=True)
    target_fieldnames = list(row.keys())

    # Use a sidecar lock file to serialize concurrent writers. Variants run in
    # parallel within one (dataset, config) combo and all append to the same
    # result CSV; without this lock the "file missing / schema migrate" branch
    # races and drops rows.
    lock_path = output_file.with_suffix(output_file.suffix + ".lock")
    with open(lock_path, "w") as lock_f:
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
        try:
            if not output_file.is_file() or output_file.stat().st_size == 0:
                with open(output_file, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=target_fieldnames)
                    writer.writeheader()
                    writer.writerow(row)
                return

            with open(output_file, "r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                existing_header = next(reader, [])

            if existing_header == target_fieldnames:
                with open(output_file, "a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=target_fieldnames)
                    writer.writerow(row)
                return

            with open(output_file, "r", newline="", encoding="utf-8") as f:
                existing_rows = list(csv.DictReader(f))

            merged_fieldnames = target_fieldnames + [
                c for c in existing_header if c not in target_fieldnames
            ]
            migrated_path = output_file.with_suffix(output_file.suffix + ".tmp")
            with open(migrated_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=merged_fieldnames)
                writer.writeheader()
                for old_row in existing_rows:
                    normalized = {k: old_row.get(k, "") for k in merged_fieldnames}
                    writer.writerow(normalized)
                writer.writerow({k: row.get(k, "") for k in merged_fieldnames})

            migrated_path.replace(output_file)
        finally:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)


def save_single_result_to_csv(
    variant,
    model_type,
    train_metrics,
    test_metrics,
    test_metrics_original,
    train_file_path,
    train_metrics_original=None,
    llm_field=None,
    llm_fields=None,
    llm_input_mode=None,
    llm_input_name=None,
    llm_responses_length=None,
    llm_vector_transform=None,
    prediction_head=None,
    result_file_name=None,
    save_split_results=True,
    result_precision=4,
):
    """
    保存单次实验结果到CSV

    命名规则:
    - 输入: dataset/Aggreated/eedi_train.json
    - 输出: results/Aggreated_eedi_result.csv (或 Aggreated_eedi_gpt-4o_norm_result.csv if llm_field specified)

    所有数字保留4位小数

    Args:
        variant: 特征变体名称
        model_type: 模型类型
        train_metrics: 训练集指标（归一化空间）
        test_metrics: 测试集指标（归一化空间）
        test_metrics_original: 测试集指标（原始空间）
        train_file_path: 训练文件路径
        train_metrics_original: 训练集指标（原始空间，可选）
        llm_field: 使用的LLM字段名称（可选）
        llm_fields: 使用的多个LLM字段名称（可选）
        llm_input_mode: LLM输入构造模式标签（可选）
        llm_input_name: LLM输入构造名称标签（可选）
        llm_responses_length: LLM responses维度（主要用于x_all_llm，可选）
        llm_vector_transform: 向量化LLM输入的规范化方式（可选）
        prediction_head: MLP prediction head and training objective（可选）
        result_file_name: 指定输出文件名（可选，写入results目录）
        save_split_results: 是否额外写入 results/train 和 results/test
        result_precision: CSV 中浮点指标保留的小数位数
    """
    train_path = Path(train_file_path)

    # 提取目录名和文件前缀
    # dataset/Aggreated/eedi_train.json -> Aggreated, eedi
    parent_dir = train_path.parent.name  # Aggreated
    file_prefix = train_path.stem.split('_')[0]  # eedi (从 eedi_train 提取 eedi)

    # 生成输出文件名
    if result_file_name:
        output_relative_path = Path(result_file_name)
        if output_relative_path.is_absolute():
            raise ValueError("result_file_name must be relative to the results/ directory.")
    elif llm_field:
        # 如果指定了 llm_field，在文件名中包含它
        output_relative_path = Path(f"{parent_dir}_{file_prefix}_{llm_field}_result.csv")
    elif llm_fields:
        llm_fields_slug = "__".join(llm_fields)
        output_relative_path = Path(f"{parent_dir}_{file_prefix}_{llm_fields_slug}_result.csv")
    else:
        output_relative_path = Path(f"{parent_dir}_{file_prefix}_result.csv")

    output_root = Path("results")
    combined_file = output_root / output_relative_path
    train_file = output_root / "train" / output_relative_path
    test_file = output_root / "test" / output_relative_path

    # Collect train metrics
    train_acc_base_hard = train_acc_model_hard = np.nan
    train_acc_base_soft = train_acc_model_soft = np.nan
    train_acc_base_mad = train_acc_model_mad = np.nan
    if train_metrics_original is not None:
        train_acc_base_hard = train_metrics_original.get('acc_baseline_hard')
        train_acc_model_hard = train_metrics_original.get('acc_model_hard')
        train_acc_base_soft = train_metrics_original.get('acc_baseline_soft')
        train_acc_model_soft = train_metrics_original.get('acc_model_soft')
        train_acc_base_mad = train_metrics_original.get('acc_baseline_mad')
        train_acc_model_mad = train_metrics_original.get('acc_model_mad')

    # Collect test metrics
    test_mse_base_norm = test_metrics.get('mse_baseline')
    test_mse_model_norm = test_metrics.get('mse')
    test_mae_base_norm = test_metrics.get('mae_baseline')
    test_mae_model_norm = test_metrics.get('mae')
    test_mse_base_original = test_metrics_original.get('mse_baseline')
    test_mse_model_original = test_metrics_original.get('mse')
    test_mae_base_original = test_metrics_original.get('mae_baseline')
    test_mae_model_original = test_metrics_original.get('mae')
    test_acc_base_hard = test_metrics_original.get('acc_baseline_hard')
    test_acc_model_hard = test_metrics_original.get('acc_model_hard')
    test_acc_base_soft = test_metrics_original.get('acc_baseline_soft')
    test_acc_model_soft = test_metrics_original.get('acc_model_soft')
    test_acc_base_mad = test_metrics_original.get('acc_baseline_mad')
    test_acc_model_mad = test_metrics_original.get('acc_model_mad')

    # Improvement metrics
    test_mae_improvement_pct = test_metrics_original.get(
        'mae_improvement_pct',
        _improvement_pct(test_mae_base_original, test_mae_model_original, higher_is_better=False),
    )
    test_mse_original_improvement_pct = test_metrics_original.get(
        'mse_improvement_pct',
        _improvement_pct(test_mse_base_original, test_mse_model_original, higher_is_better=False),
    )
    test_mse_norm_improvement_pct = _improvement_pct(
        test_mse_base_norm, test_mse_model_norm, higher_is_better=False
    )
    test_mae_norm_improvement_pct = _improvement_pct(
        test_mae_base_norm, test_mae_model_norm, higher_is_better=False
    )
    test_acc_improvement_pct = _improvement_pct(
        test_acc_base_hard, test_acc_model_hard, higher_is_better=True
    )
    test_acc_soft_improvement_pct = _improvement_pct(
        test_acc_base_soft, test_acc_model_soft, higher_is_better=True
    )
    test_acc_mad_improvement_pct = _improvement_pct(
        test_acc_base_mad, test_acc_model_mad, higher_is_better=True
    )
    train_acc_improvement_pct = _improvement_pct(
        train_acc_base_hard, train_acc_model_hard, higher_is_better=True
    )
    train_acc_soft_improvement_pct = _improvement_pct(
        train_acc_base_soft, train_acc_model_soft, higher_is_better=True
    )
    train_acc_mad_improvement_pct = _improvement_pct(
        train_acc_base_mad, train_acc_model_mad, higher_is_better=True
    )

    # Build rows with adjacent base/model/improvement columns for easier comparison.
    common_fields = {
        "variant": variant,
        "model_type": model_type,
        "llm_input_mode": llm_input_mode,
        "llm_input_name": llm_input_name,
        "llm_responses_length": llm_responses_length,
        "llm_vector_transform": llm_vector_transform,
        "prediction_head": prediction_head,
        "llm_field": llm_field,
        "llm_fields": "|".join(llm_fields) if llm_fields else None,
    }

    train_row = {
        **common_fields,
        "train_mse_norm": train_metrics.get('mse'),
        "train_mae_norm": train_metrics.get('mae'),
        "train_r2_norm": train_metrics.get('r2'),
        "train_acc_base_hard": train_acc_base_hard,
        "train_acc_model_hard": train_acc_model_hard,
        "train_acc_improvement_pct": train_acc_improvement_pct,
        "train_acc_base_soft": train_acc_base_soft,
        "train_acc_model_soft": train_acc_model_soft,
        "train_acc_soft_improvement_pct": train_acc_soft_improvement_pct,
        "train_acc_base_mad": train_acc_base_mad,
        "train_acc_model_mad": train_acc_model_mad,
        "train_acc_mad_improvement_pct": train_acc_mad_improvement_pct,
    }

    test_row = {
        **common_fields,
        "test_r2_norm": test_metrics.get('r2'),
        "test_r2_original": test_metrics_original.get('r2'),
        "test_mse_base_norm": test_mse_base_norm,
        "test_mse_model_norm": test_mse_model_norm,
        "test_mse_norm_improvement_pct": test_mse_norm_improvement_pct,
        "test_mae_base_norm": test_mae_base_norm,
        "test_mae_model_norm": test_mae_model_norm,
        "test_mae_norm_improvement_pct": test_mae_norm_improvement_pct,
        "test_mse_base_original": test_mse_base_original,
        "test_mse_model_original": test_mse_model_original,
        "test_mse_original_improvement_pct": test_mse_original_improvement_pct,
        "test_mae_base_original": test_mae_base_original,
        "test_mae_model_original": test_mae_model_original,
        "test_mae_improvement_pct": test_mae_improvement_pct,
        "test_acc_base_hard": test_acc_base_hard,
        "test_acc_model_hard": test_acc_model_hard,
        "test_acc_improvement_pct": test_acc_improvement_pct,
        "test_acc_base_soft": test_acc_base_soft,
        "test_acc_model_soft": test_acc_model_soft,
        "test_acc_soft_improvement_pct": test_acc_soft_improvement_pct,
        "test_acc_base_mad": test_acc_base_mad,
        "test_acc_model_mad": test_acc_model_mad,
        "test_acc_mad_improvement_pct": test_acc_mad_improvement_pct,
        # Extended reporting metrics (model + baseline LLM).
        "test_pearson_r_base": test_metrics_original.get('baseline_pearson_r'),
        "test_pearson_r_model": test_metrics_original.get('model_pearson_r'),
        "test_spearman_r_base": test_metrics_original.get('baseline_spearman_r'),
        "test_spearman_r_model": test_metrics_original.get('model_spearman_r'),
        "test_fisher_z_base": test_metrics_original.get('baseline_fisher_z'),
        "test_fisher_z_model": test_metrics_original.get('model_fisher_z'),
        "test_variance_ratio_base": test_metrics_original.get('baseline_variance_ratio'),
        "test_variance_ratio_model": test_metrics_original.get('model_variance_ratio'),
        "test_glass_delta_base": test_metrics_original.get('baseline_glass_delta'),
        "test_glass_delta_model": test_metrics_original.get('model_glass_delta'),
        "test_per_person_pearson_base": test_metrics_original.get('baseline_per_person_pearson'),
        "test_per_person_pearson_model": test_metrics_original.get('model_per_person_pearson'),
        "test_per_person_spearman_base": test_metrics_original.get('baseline_per_person_spearman'),
        "test_per_person_spearman_model": test_metrics_original.get('model_per_person_spearman'),
        "test_per_person_groups_used": test_metrics_original.get('model_per_person_groups_used'),
        "test_individuation_gap_base": test_metrics_original.get('baseline_individuation_gap'),
        "test_individuation_gap_model": test_metrics_original.get('model_individuation_gap'),
        "test_range_coverage_ratio_base": test_metrics_original.get('baseline_range_coverage_ratio'),
        "test_range_coverage_ratio_model": test_metrics_original.get('model_range_coverage_ratio'),
    }

    combined_row = {**train_row, **test_row}

    if not isinstance(result_precision, int) or not 1 <= result_precision <= 17:
        raise ValueError("result_precision must be an integer from 1 through 17")

    train_row_fmt = {
        k: _format_metric_value(v, result_precision) for k, v in train_row.items()
    }
    test_row_fmt = {
        k: _format_metric_value(v, result_precision) for k, v in test_row.items()
    }
    combined_row_fmt = {
        k: _format_metric_value(v, result_precision) for k, v in combined_row.items()
    }

    # Save files
    _append_row_to_csv(combined_file, combined_row_fmt)

    print(f"Results saved to: {combined_file}")
    if save_split_results:
        _append_row_to_csv(train_file, train_row_fmt)
        _append_row_to_csv(test_file, test_row_fmt)
        print(f"Train metrics saved to: {train_file}")
        print(f"Test metrics saved to: {test_file}")
