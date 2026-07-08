import type { ArtifactReferences, ReproducibilityMetadata } from '../api/types'

interface DebugDetailsProps {
  artifacts: ArtifactReferences
  reproducibility: ReproducibilityMetadata
}

function MonoBlock({ children }: { children: string }) {
  return <dd className="mono mono-block">{children}</dd>
}

export function DebugDetails({ artifacts, reproducibility }: DebugDetailsProps) {
  return (
    <details className="debug-details tech-details">
      <summary>Debug / artifact details</summary>
      <div className="tech-details-body debug-grid">
        <div>
          <h4>Artifacts</h4>
          <dl className="debug-dl">
            <dt>model_dataset_id</dt>
            <MonoBlock>{artifacts.model_dataset_id}</MonoBlock>
            <dt>baseline_run_id</dt>
            <MonoBlock>{artifacts.baseline_run_id}</MonoBlock>
            <dt>threshold_policy_id</dt>
            <MonoBlock>{artifacts.threshold_policy_id}</MonoBlock>
            <dt>abstention_policy_id</dt>
            <MonoBlock>{artifacts.abstention_policy_id}</MonoBlock>
            <dt>retrieval_run_id</dt>
            <MonoBlock>{artifacts.retrieval_run_id}</MonoBlock>
          </dl>
        </div>
        <div>
          <h4>Reproducibility</h4>
          <dl className="debug-dl">
            <dt>inference_config_path</dt>
            <MonoBlock>{reproducibility.inference_config_path}</MonoBlock>
            <dt>model_semantic_sha256</dt>
            <MonoBlock>{reproducibility.model_semantic_sha256}</MonoBlock>
            <dt>index_semantic_sha256</dt>
            <MonoBlock>{reproducibility.index_semantic_sha256}</MonoBlock>
            <dt>baseline_experiment_sha256</dt>
            <MonoBlock>{reproducibility.baseline_experiment_sha256}</MonoBlock>
            <dt>numerical_environment_sha256</dt>
            <MonoBlock>{reproducibility.numerical_environment_sha256}</MonoBlock>
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
