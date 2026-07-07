export type AbstentionReason =
  | 'no_labels_predicted'
  | 'confidence_below_threshold'
  | 'confidence_meets_threshold'

export type InferenceWarning = 'empty_title' | 'empty_body' | 'no_labels_predicted'

export type ReviewAction = 'accepted' | 'corrected' | 'rejected'

export interface HealthResponse {
  status: 'ok'
  schema_version: '1'
  repository: string
  inference_config_path: string
}

export interface InferRequest {
  title: string
  body?: string
  top_k?: number | null
  issue_number?: number | null
  issue_id?: number | null
}

export interface LabelScore {
  label: string
  score: number
}

export interface PredictedLabel {
  label: string
  score: number
}

export interface InferenceInputSummary {
  title: string
  body_preview: string
  feature_text_sha256: string
  text_representation_version: string
}

export interface ClassificationResult {
  label_order: string[]
  scores: LabelScore[]
  threshold: number
  threshold_basis_points: number
  predicted_labels: PredictedLabel[]
}

export interface AbstentionResult {
  confidence_method: 'max_predicted_label_score'
  confidence: number | null
  threshold: number
  threshold_basis_points: number
  should_abstain: boolean
  reason: AbstentionReason
}

export interface SimilarIssueResult {
  rank: number
  issue_id: number
  issue_number: number
  similarity: number
  neighbor_selected_labels: string[]
  predicted_label_overlap: string[]
}

export interface RetrievalResult {
  method: 'tfidf_cosine'
  top_k: number
  similar_issues: SimilarIssueResult[]
}

export interface ArtifactReferences {
  model_dataset_id: string
  baseline_run_id: string
  threshold_policy_id: string
  abstention_policy_id: string
  retrieval_run_id: string
}

export interface ReproducibilityMetadata {
  inference_config_path: string
  model_semantic_sha256: string
  index_semantic_sha256: string
  baseline_experiment_sha256: string
  numerical_environment_sha256: string
  serialization_security_warning: string | null
}

export interface InferenceResponse {
  schema_version: '1'
  repository: string
  generated_at: string
  input: InferenceInputSummary
  classification: ClassificationResult
  abstention: AbstentionResult
  retrieval: RetrievalResult
  artifacts: ArtifactReferences
  reproducibility: ReproducibilityMetadata
  warnings: InferenceWarning[]
}

export interface InferenceArtifactsInput {
  model_dataset_id: string
  baseline_run_id: string
  threshold_policy_id: string
  abstention_policy_id: string
  retrieval_run_id: string
}

export interface FeedbackRequest {
  feedback_schema_version?: '1'
  repository: string
  issue_number: number
  issue_title: string
  issue_body_preview?: string
  predicted_labels: string[]
  accepted_labels: string[]
  rejected_labels?: string[]
  review_action: ReviewAction
  reviewer_note?: string | null
  inference_artifacts: InferenceArtifactsInput
}

export interface FeedbackResponse {
  feedback_id: string
  created_at: string
  status: 'stored'
}
