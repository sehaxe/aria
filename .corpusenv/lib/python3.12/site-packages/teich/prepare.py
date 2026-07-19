from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import Dataset, concatenate_datasets

from .formatter import PrepareReport, format_data, normalize_prepared_dataset_features
from .loader import _resolve_hf_token, load_traces


_DATASET_MIX_SEED = 3407


@dataclass(slots=True)
class _SourceMixEntry:
    name: str
    source: str | Path | Dataset
    max_examples: int | None
    percentage: float | None
    has_explicit_mix_value: bool
    chat_template_kwargs: dict[str, Any] | None


@dataclass(slots=True)
class _ResolvedSourceMix:
    datasets: list[Dataset]
    probabilities: list[float]
    max_examples: int | None
    names: list[str]
    rigid_percentages: bool
    chat_template_kwargs: list[dict[str, Any] | None]


@dataclass(slots=True)
class _ResolvedSourceList:
    datasets: list[Dataset]
    max_examples: int | None
    names: list[str]


def prepare_data(
    source_or_dataset: str | Path | Dataset | Mapping[str, Any] | Sequence[str | Path | Dataset | Mapping[str, Any]],
    tokenizer: Any,
    *,
    split: str | None = "train",
    revision: str | None = None,
    token: str | None = None,
    hf_token: str | None = None,
    cache_dir: str | Path | None = None,
    local_dir: str | Path | None = None,
    max_examples: int | None = None,
    messages_column: str = "messages",
    tools_column: str = "tools",
    text_column: str = "text",
    chat_template_kwargs: dict[str, Any] | None = None,
    train_on_reasoning: bool | None = None,
    teich_masking: bool = True,
    max_length: int | None = None,
    oversized_policy: str | None = None,
    drop_oversized_examples: bool = True,
    trim_oversized_followups: bool = False,
    preserve_columns: bool | Sequence[str] | None = None,
    tokenize: bool = False,
    validate_tools: bool = False,
    strict: bool = True,
    return_report: bool = False,
    verbose: bool = True,
) -> Dataset | tuple[Dataset, PrepareReport]:
    effective_token = _resolve_hf_token(token, hf_token)
    report = PrepareReport() if return_report else None
    dataset = _resolve_source_dataset(
        source_or_dataset,
        split=split,
        revision=revision,
        token=effective_token,
        cache_dir=cache_dir,
        local_dir=local_dir,
        max_examples=max_examples,
    )
    if isinstance(dataset, _ResolvedSourceMix):
        formatted_datasets = [
            format_data(
                source_dataset,
                tokenizer,
                messages_column=messages_column,
                tools_column=tools_column,
                text_column=text_column,
                chat_template_kwargs=_merge_chat_template_kwargs(chat_template_kwargs, source_chat_template_kwargs),
                train_on_reasoning=train_on_reasoning,
                teich_masking=teich_masking,
                max_length=max_length,
                oversized_policy=oversized_policy,
                drop_oversized_examples=drop_oversized_examples,
                trim_oversized_followups=trim_oversized_followups,
                preserve_columns=preserve_columns,
                source_key=source_name,
                report=report,
                validate_tools=validate_tools,
                tokenize=tokenize,
                strict=strict,
                verbose=verbose,
            )
            for source_dataset, source_chat_template_kwargs, source_name in zip(
                dataset.datasets,
                dataset.chat_template_kwargs,
                dataset.names,
                strict=True,
            )
        ]
        prepared = _mix_prepared_datasets(
            formatted_datasets,
            probabilities=dataset.probabilities,
            max_examples=dataset.max_examples,
            rigid_percentages=dataset.rigid_percentages,
        )
        if report is not None:
            report.returned_rows = prepared.num_rows
            return prepared, report
        return prepared
    if isinstance(dataset, _ResolvedSourceList):
        formatted_datasets = [
            format_data(
                source_dataset,
                tokenizer,
                messages_column=messages_column,
                tools_column=tools_column,
                text_column=text_column,
                chat_template_kwargs=chat_template_kwargs,
                train_on_reasoning=train_on_reasoning,
                teich_masking=teich_masking,
                max_length=max_length,
                oversized_policy=oversized_policy,
                drop_oversized_examples=drop_oversized_examples,
                trim_oversized_followups=trim_oversized_followups,
                preserve_columns=preserve_columns,
                source_key=source_name,
                report=report,
                validate_tools=validate_tools,
                tokenize=tokenize,
                strict=strict,
                verbose=verbose,
            )
            for source_dataset, source_name in zip(dataset.datasets, dataset.names, strict=True)
        ]
        formatted = concatenate_datasets(
            [normalize_prepared_dataset_features(formatted_dataset) for formatted_dataset in formatted_datasets]
        )
        if dataset.max_examples is None:
            if report is not None:
                report.returned_rows = formatted.num_rows
                return formatted, report
            return formatted
        limit = min(dataset.max_examples, formatted.num_rows)
        prepared = formatted.shuffle(seed=_DATASET_MIX_SEED).select(range(limit))
        if report is not None:
            report.returned_rows = prepared.num_rows
            return prepared, report
        return prepared
    prepared = format_data(
        dataset,
        tokenizer,
        messages_column=messages_column,
        tools_column=tools_column,
        text_column=text_column,
        chat_template_kwargs=chat_template_kwargs,
        train_on_reasoning=train_on_reasoning,
        teich_masking=teich_masking,
        max_length=max_length,
        oversized_policy=oversized_policy,
        drop_oversized_examples=drop_oversized_examples,
        trim_oversized_followups=trim_oversized_followups,
        preserve_columns=preserve_columns,
        report=report,
        validate_tools=validate_tools,
        tokenize=tokenize,
        strict=strict,
        verbose=verbose,
    )
    if report is not None:
        report.returned_rows = prepared.num_rows
        return prepared, report
    return prepared


def _resolve_source_dataset(
    source_or_dataset: str | Path | Dataset | Mapping[str, Any] | Sequence[str | Path | Dataset | Mapping[str, Any]],
    *,
    split: str | None,
    revision: str | None,
    token: str | None,
    cache_dir: str | Path | None,
    local_dir: str | Path | None,
    max_examples: int | None,
) -> Dataset | _ResolvedSourceList | _ResolvedSourceMix:
    if isinstance(source_or_dataset, Dataset):
        return _resolve_single_source_dataset(
            source_or_dataset,
            split=split,
            revision=revision,
            token=token,
            cache_dir=cache_dir,
            local_dir=local_dir,
            max_examples=max_examples,
        )
    if isinstance(source_or_dataset, Mapping):
        entries, mix_max_examples = _source_mix_entries_from_mapping(source_or_dataset, max_examples)
        return _resolve_source_mix(
            entries,
            split=split,
            revision=revision,
            token=token,
            cache_dir=cache_dir,
            local_dir=local_dir,
            max_examples=mix_max_examples,
        )
    if isinstance(source_or_dataset, Sequence) and not isinstance(source_or_dataset, (str, bytes, bytearray)):
        sources = list(source_or_dataset)
        if not sources:
            raise ValueError("At least one dataset must be provided.")
        if any(isinstance(source, Mapping) for source in sources):
            entries = [_source_mix_entry_from_value(source, default_name=f"source_{index}") for index, source in enumerate(sources)]
            return _resolve_source_mix(
                entries,
                split=split,
                revision=revision,
                token=token,
                cache_dir=cache_dir,
                local_dir=local_dir,
                max_examples=max_examples,
            )
        if max_examples is not None and max_examples < 0:
            raise ValueError("max_examples must be non-negative.")
        return _ResolvedSourceList(
            datasets=[
                _resolve_single_source_dataset(
                    source,
                    split=split,
                    revision=revision,
                    token=token,
                    cache_dir=cache_dir,
                    local_dir=local_dir,
                    max_examples=None,
                )
                for source in sources
            ],
            max_examples=max_examples,
            names=[f"source_{index}" for index in range(len(sources))],
        )
    return _resolve_single_source_dataset(
        source_or_dataset,
        split=split,
        revision=revision,
        token=token,
        cache_dir=cache_dir,
        local_dir=local_dir,
        max_examples=max_examples,
    )


def _source_mix_entries_from_mapping(
    source_mix: Mapping[str, Any],
    max_examples: int | None,
) -> tuple[list[_SourceMixEntry], int | None]:
    if "sources" in source_mix:
        sources = source_mix["sources"]
        if not isinstance(sources, Sequence) or isinstance(sources, (str, bytes, bytearray)):
            raise TypeError("A source mix 'sources' value must be a sequence of sources or source configuration objects.")
        mix_max_examples = _optional_non_negative_int(source_mix.get("max_examples"), "max_examples")
        entries = [_source_mix_entry_from_value(source, default_name=f"source_{index}") for index, source in enumerate(sources)]
        return entries, max_examples if max_examples is not None else mix_max_examples
    if _mapping_is_source_entry(source_mix):
        return [_source_mix_entry_from_value(source_mix, default_name="source_0")], max_examples
    mix_max_examples = _optional_non_negative_int(source_mix.get("max_examples"), "max_examples")
    entries = [
        _source_mix_entry_from_value(value, default_name=str(name))
        for name, value in source_mix.items()
        if name != "max_examples"
    ]
    if not entries:
        raise ValueError("At least one dataset must be provided.")
    return entries, max_examples if max_examples is not None else mix_max_examples


def _source_mix_entry_from_value(value: Any, *, default_name: str) -> _SourceMixEntry:
    if isinstance(value, Mapping):
        if _mapping_is_source_entry(value):
            source = value.get("source", value.get("dataset", value.get("path")))
            name_value = value.get("name", default_name)
            if not isinstance(name_value, str) or not name_value:
                raise TypeError("A source mix entry name must be a non-empty string.")
            return _SourceMixEntry(
                name=name_value,
                source=_validate_source_value(source),
                max_examples=_optional_non_negative_int(value.get("max_examples"), f"{name_value}.max_examples"),
                percentage=_optional_mix_value(value, name_value),
                has_explicit_mix_value=_has_explicit_mix_value(value),
                chat_template_kwargs=_optional_chat_template_kwargs(
                    value.get("chat_template_kwargs"),
                    f"{name_value}.chat_template_kwargs",
                ),
            )
        raise TypeError("A source mix entry mapping must include a 'source', 'dataset', or 'path' key.")
    return _SourceMixEntry(
        name=default_name,
        source=_validate_source_value(value),
        max_examples=None,
        percentage=None,
        has_explicit_mix_value=False,
        chat_template_kwargs=None,
    )


def _mapping_is_source_entry(value: Mapping[str, Any]) -> bool:
    return any(key in value for key in ("source", "dataset", "path"))


def _validate_source_value(value: Any) -> str | Path | Dataset:
    if isinstance(value, (str, Path, Dataset)):
        return value
    raise TypeError("A source mix entry must reference a dataset path, Hugging Face dataset ID, or datasets.Dataset object.")


def _optional_non_negative_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return value


def _optional_mix_value(value: Mapping[str, Any], source_name: str) -> float | None:
    if "weight" in value:
        return _optional_positive_float(value["weight"], f"{source_name}.weight")
    if "percentage" in value:
        return _optional_percentage(value["percentage"], f"{source_name}.percentage")
    if "proportion" in value:
        return _optional_percentage(value["proportion"], f"{source_name}.proportion")
    return None


def _has_explicit_mix_value(value: Mapping[str, Any]) -> bool:
    return any(key in value and value[key] is not None for key in ("weight", "percentage", "proportion"))


def _optional_positive_float(value: Any, name: str) -> float | None:
    if value is None:
        return None
    if not isinstance(value, int | float) or value <= 0:
        raise ValueError(f"{name} must be a positive number.")
    return float(value)


def _optional_percentage(value: Any, name: str) -> float | None:
    percentage = _optional_positive_float(value, name)
    if percentage is None:
        return None
    if percentage > 100:
        raise ValueError(f"{name} must be between 0 and 100 when specified as a percentage.")
    return percentage / 100 if percentage > 1 else percentage


def _optional_chat_template_kwargs(value: Any, name: str) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping of apply_chat_template keyword arguments.")
    if not all(isinstance(key, str) for key in value):
        raise TypeError(f"{name} keys must be strings.")
    return dict(value)


def _merge_chat_template_kwargs(
    global_kwargs: dict[str, Any] | None,
    source_kwargs: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if source_kwargs is None:
        return global_kwargs
    merged = dict(global_kwargs or {})
    merged.update(source_kwargs)
    return merged


def _resolve_source_mix(
    entries: Sequence[_SourceMixEntry],
    *,
    split: str | None,
    revision: str | None,
    token: str | None,
    cache_dir: str | Path | None,
    local_dir: str | Path | None,
    max_examples: int | None,
) -> _ResolvedSourceMix:
    if not entries:
        raise ValueError("At least one dataset must be provided.")
    if max_examples is not None and max_examples < 0:
        raise ValueError("max_examples must be non-negative.")
    source_percentages = [entry.percentage for entry in entries]
    if all(percentage is None for percentage in source_percentages):
        probabilities = [1.0 / len(entries)] * len(entries)
    elif any(percentage is None for percentage in source_percentages):
        specified_total = sum(percentage or 0.0 for percentage in source_percentages)
        if specified_total >= 1.0:
            raise ValueError("Source mix percentages must leave room for entries without an explicit percentage.")
        unspecified_count = sum(1 for percentage in source_percentages if percentage is None)
        fallback_percentage = (1.0 - specified_total) / unspecified_count
        probabilities = [percentage if percentage is not None else fallback_percentage for percentage in source_percentages]
    else:
        total = sum(percentage or 0.0 for percentage in source_percentages)
        probabilities = [(percentage or 0.0) / total for percentage in source_percentages]
    datasets = [
        _resolve_single_source_dataset(
            entry.source,
            split=split,
            revision=revision,
            token=token,
            cache_dir=cache_dir,
            local_dir=local_dir,
            max_examples=entry.max_examples,
        )
        for entry in entries
    ]
    return _ResolvedSourceMix(
        datasets=datasets,
        probabilities=probabilities,
        max_examples=max_examples,
        names=[entry.name for entry in entries],
        rigid_percentages=any(entry.has_explicit_mix_value for entry in entries),
        chat_template_kwargs=[entry.chat_template_kwargs for entry in entries],
    )


def _mix_prepared_datasets(
    datasets: Sequence[Dataset],
    *,
    probabilities: Sequence[float],
    max_examples: int | None,
    rigid_percentages: bool = False,
) -> Dataset:
    if not datasets:
        raise ValueError("At least one dataset must be provided.")
    if len(datasets) != len(probabilities):
        raise ValueError("Dataset mix probabilities must match the number of datasets.")
    available_counts = [dataset.num_rows for dataset in datasets]
    if any(count == 0 for count in available_counts):
        raise ValueError("Dataset mix sources must each contain at least one formatted training example.")
    target_total = min(max_examples, sum(available_counts)) if max_examples is not None else sum(available_counts)
    if target_total == 0:
        return datasets[0].select(range(0))
    if not rigid_percentages:
        combined = concatenate_datasets([normalize_prepared_dataset_features(dataset) for dataset in datasets])
        shuffled = combined.shuffle(seed=_DATASET_MIX_SEED)
        return shuffled.select(range(target_total))
    source_counts = _allocate_rigid_source_counts(available_counts, probabilities, target_total)
    selected_datasets = [
        normalize_prepared_dataset_features(dataset.shuffle(seed=_DATASET_MIX_SEED + index).select(range(count)))
        for index, (dataset, count) in enumerate(zip(datasets, source_counts, strict=True))
        if count > 0
    ]
    if not selected_datasets:
        return datasets[0].select(range(0))
    return concatenate_datasets(selected_datasets).shuffle(seed=_DATASET_MIX_SEED)


def _allocate_rigid_source_counts(
    available_counts: Sequence[int],
    probabilities: Sequence[float],
    target_total: int,
) -> list[int]:
    upper_bound = min(
        target_total,
        *[int((available + 1 - 1e-12) / probability) for available, probability in zip(available_counts, probabilities, strict=True)],
    )
    for total in range(upper_bound, -1, -1):
        counts = _largest_remainder_counts(probabilities, total)
        if all(count <= available for count, available in zip(counts, available_counts, strict=True)):
            return counts
    return [0] * len(available_counts)


def _largest_remainder_counts(probabilities: Sequence[float], target_total: int) -> list[int]:
    desired_counts = [target_total * probability for probability in probabilities]
    quotas = [int(desired) for desired in desired_counts]
    remaining = target_total - sum(quotas)
    fractional_parts = [
        (desired - quota, index)
        for index, (desired, quota) in enumerate(zip(desired_counts, quotas, strict=True))
    ]
    for _, index in sorted(fractional_parts, key=lambda item: (-item[0], item[1])):
        if remaining <= 0:
            break
        quotas[index] += 1
        remaining -= 1
    return quotas


def _resolve_single_source_dataset(
    source_or_dataset: str | Path | Dataset,
    *,
    split: str | None,
    revision: str | None,
    token: str | None,
    cache_dir: str | Path | None,
    local_dir: str | Path | None,
    max_examples: int | None,
) -> Dataset:
    if isinstance(source_or_dataset, Dataset):
        if max_examples is None:
            return source_or_dataset
        if max_examples < 0:
            raise ValueError("max_examples must be non-negative.")
        limit = min(max_examples, source_or_dataset.num_rows)
        return source_or_dataset.shuffle(seed=3407).select(range(limit))
    if not isinstance(source_or_dataset, (str, Path)):
        raise TypeError(
            "A sequence source must contain only dataset paths, Hugging Face dataset IDs, or datasets.Dataset objects."
        )
    return load_traces(
        source_or_dataset,
        split=split,
        revision=revision,
        token=token,
        cache_dir=cache_dir,
        local_dir=local_dir,
        max_examples=max_examples,
    )
