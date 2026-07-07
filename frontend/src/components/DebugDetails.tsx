import type { ArtifactReferences, ReproducibilityMetadata } from '../api/types'

interface DebugDetailsProps {
  artifacts: ArtifactReferences
  reproducibility: ReproducibilityMetadata
}

export function DebugDetails({ artifacts, reproducibility }: DebugDetailsProps) {
  return (
    <details className="debug-details">
      <summary>Debug / artifact details</summary>
      <div className="debug-grid">
        <div>
          <h4>Artifacts</h4>
          <dl>
            <dt>model_dataset_id</dt>
            <dd>{artifacts.model_dataset_id}</dd>
            <dt>baseline_run_id</dt>
            <dd>{artifacts.baseline_run_id}</dd>
            <dt>threshold_policy_id</dt>
            <dd>{artifacts.threshold_policy_id}</dd>
            <dt>abstention_policy_id</dt>
            <dd>{artifacts.abstention_policy_id}</dd>
            <dt>retrieval_run_id</dt>
            <dd>{artifacts.retrieval_run_id}</dd>
          </dl>
        </div>
        <div>
          <h4>Reproducibility</h4>
          <dl>
            <dt>inference_config_path</dt>
            <dd>{reproducibility.inference_config_path}</dd>
            <dt>model_semantic_sha256</dt>
            <dd className="mono">{reproducibility.model_semantic_sha256}</dd>
            <dt>index_semantic_sha256</dt>
            <dd className="mono">{reproducibility.index_semantic_sha256}</dd>
            <dt>baseline_experiment_sha256</dt>
            <dd className="mono">{reproducibility.baseline_experiment_sha256}</dd>
            <dt>numerical_environment_sha256</dt>
            <dd className="mono">{reproducibility.numerical_environment_sha256}</dd>
            {reproducibility.serialization_security_warning && (
              <>
                <dt>serialization_security_warning</dt>
                <dd>{reproducibility.serialization_security_warning}</dd>
              </>
            )}
          </dl>
        </div>
      </div>
    </details>
  )
}
