"""Tests for model-dataset identity and model validators."""

from repotriage.model_dataset.models import (
    LABEL_MAP_SCHEMA_VERSION,
    MODEL_DATASET_OUTPUT_CONTRACTS,
    MODEL_DATASET_VERSION,
    MODEL_READY_RECORD_SCHEMA_VERSION,
    SPLIT_REPORT_SCHEMA_VERSION,
    TEMPORAL_SPLITTER_VERSION,
    TEXT_REPRESENTATION_VERSION,
    LabelMap,
    compute_model_dataset_id,
    compute_model_dataset_input_sha256,
)


def test_model_dataset_v1_output_contract_mapping() -> None:
    assert MODEL_DATASET_OUTPUT_CONTRACTS[MODEL_DATASET_VERSION] == {
        "record_schema_version": MODEL_READY_RECORD_SCHEMA_VERSION,
        "label_map_schema_version": LABEL_MAP_SCHEMA_VERSION,
        "split_report_schema_version": SPLIT_REPORT_SCHEMA_VERSION,
        "text_representation_version": TEXT_REPRESENTATION_VERSION,
        "temporal_splitter_version": TEMPORAL_SPLITTER_VERSION,
    }


def test_same_inputs_same_id() -> None:
    kwargs = dict(
        model_dataset_version=MODEL_DATASET_VERSION,
        dataset_id="20260628T161306010651Z-n1-074402d21505",
        dataset_output_sha256="a" * 64,
        policy_id="20260628T161306010651Z-n1-074402d21505-lp2-95899f0f5b37",
        policy_json_sha256="b" * 64,
        text_representation_version=TEXT_REPRESENTATION_VERSION,
        temporal_splitter_version=TEMPORAL_SPLITTER_VERSION,
        split_config_schema_version="1",
        split_config_sha256="c" * 64,
    )
    input_hash = compute_model_dataset_input_sha256(**kwargs)
    model_id = compute_model_dataset_id(kwargs["dataset_id"], input_hash)
    assert model_id == compute_model_dataset_id(kwargs["dataset_id"], input_hash)
    assert input_hash == compute_model_dataset_input_sha256(**kwargs)


def test_changing_text_version_changes_id() -> None:
    base = dict(
        model_dataset_version=MODEL_DATASET_VERSION,
        dataset_id="20260628T161306010651Z-n1-074402d21505",
        dataset_output_sha256="a" * 64,
        policy_id="20260628T161306010651Z-n1-074402d21505-lp2-95899f0f5b37",
        policy_json_sha256="b" * 64,
        temporal_splitter_version=TEMPORAL_SPLITTER_VERSION,
        split_config_schema_version="1",
        split_config_sha256="c" * 64,
    )
    h1 = compute_model_dataset_input_sha256(**base, text_representation_version="1")
    h2 = compute_model_dataset_input_sha256(**base, text_representation_version="2")
    assert h1 != h2


def test_label_map_invariants() -> None:
    label_map = LabelMap(
        policy_id="20260628T161306010651Z-n1-074402d21505-lp2-95899f0f5b37",
        target_count=2,
        labels=["Bug", "Docs"],
        label_to_index={"Bug": 0, "Docs": 1},
    )
    assert label_map.labels == ["Bug", "Docs"]
