```markdown
# CancerAI v12 — Production-Grade Multimodal Oncology AI

**Multimodal decision-support scaffold for whole-slide pathology imaging + genomics**

## Clinical Disclaimer

> ⚠️ **This system is intended as a research and decision-support scaffold only.**  
> It is **not FDA-cleared, CE-marked, or approved for direct clinical use**.  
> All outputs **must** be reviewed and validated by qualified medical professionals.  
> Never use this system as a substitute for clinical judgment.

**CancerAI v12 is explicitly NOT intended for autonomous diagnosis, treatment planning, or any clinical decision-making without human oversight.**

---

## What It Is Intended For

CancerAI v12 is a **research-grade multimodal oncology AI platform** designed to assist pathologists and multidisciplinary teams (MDTs) by:

- Analysing **Whole Slide Images (WSI)** from pathology slides (H&E stained)
- Integrating **genomic features** (128-dimensional input vector)
- Producing:
  - Cancer probability (binary classification)
  - Treatment recommendation (from 6 standard options)
  - Quantitative uncertainty estimates (MC Dropout + combined uncertainty)
  - Explainability artefacts (top-K attended tiles with coordinates, SHAP on fusion layer, Integrated Gradients on CancerHead)
  - Clinical rule-based overrides and notes (e.g., “escalate watchful waiting”, “recommend PD-L1 IHC”)
- Supporting **reject-option classification** (defer to human when uncertainty exceeds threshold)

**Primary use cases**:
- Research studies exploring multimodal AI in oncology
- Decision-support tool during MDT meetings
- Pathology workflow augmentation (highlighting high-attention tiles)
- Clinical rule prototyping and testing (hot-reloadable rules engine)
- Training and validation of new foundation-model backbones

**Not intended for**:
- Primary diagnosis
- Real-time intraoperative use
- Any deployment without IRB/ethics approval and prospective validation

---

## Key Features (v12 Highlights)

| Feature                        | Description                                                                 | Benefit |
|--------------------------------|-----------------------------------------------------------------------------|---------|
| **ONNX Production Runtime**    | Full export of all components; lean inference image (~3-4× smaller)        | Faster, GPU/CPU/TensorRT ready, no TensorFlow in prod |
| **Multivariate Drift Detection** | MMD on full embeddings + per-dimension alerts + image quality stats       | Early warning of distribution shift |
| **Tile Quality Filtering**     | Laplacian variance, brightness, saturation-std artefact rejection          | Higher prediction reliability |
| **Reject-Option Classification** | Configurable uncertainty threshold → `deferred_to_human`                 | Safety-first design |
| **Hot-Reloadable Clinical Rules** | External YAML rules with priority, actions (`set_treatment`, `add_note`, `recommend_test`) | Clinicians update logic without redeploy |
| **Explainability**             | Top-K attended tiles + coordinates, SHAP on fusion, IG on CancerHead      | Pathologist trust & review |
| **RBAC + Per-User Rate Limits**| Roles: `viewer`, `mdt_reviewer`, `admin` + OIDC/LDAP/DB auth             | Enterprise-grade security |
| **Request-ID Correlation**     | Full trace across logs, metrics, audit                                     | Observability |
| **Mixed-Precision Training**   | Optional distributed training (MirroredStrategy)                           | Faster training on multi-GPU |

---

## Architecture

CancerAI follows a **modular, componentised pipeline**:

```
WSI Input
   ↓ (OpenSlide + quality filtering)
Tile Extraction → TileEncoder (UNI backbone) → AttentionMIL (hierarchical)
   ↓
Slide Embedding (with MC uncertainty)
   ↓
[Image Pathway]                [Genomic Pathway]
CancerHead (MC Dropout)   →   GenomicTreatmentModel (MC Dropout)
   ↓                                 ↓
         CrossAttentionFusion (image + genomic)
                  ↓
            Final Cancer Probability + Treatment Recommendation
                  ↓
         Clinical Rules Engine (priority-ordered, versioned YAML)
                  ↓
   Output: prob, treatment, uncertainty, top-K tiles, SHAP/IG, notes, risk_level
```

**Inference modes**:
- **Production**: ONNX Runtime (`use_onnx: true`)
- **Training/Dev**: Keras/TensorFlow

**Uncertainty propagation**: MC Dropout at every stage → combined uncertainty used for reject-option and rules.

---

## Repository Structure

```
cancerAI_v12/
├── README.md                          ← You are reading this
├── Dockerfile                         ← Full training + inference image
├── Dockerfile.inference               ← Lean ONNX-only production image
├── docker-compose.yml
├── config/
│   ├── config.yaml
│   └── rules/rules_v1.yaml            ← Hot-reloadable clinical rules
├── app/
│   ├── main.py                        ← FastAPI entrypoint
│   ├── auth.py, middleware.py
│   ├── models/                        ← TileEncoder, MIL, CancerHead, Fusion, Treatment
│   ├── pipelines/                     ← WSI tiling + quality filter, genomics
│   ├── inference/                     ← Predictor, OnnxModelRunner, Explainer
│   ├── rules/engine.py                ← Priority-aware rules DSL
│   ├── monitoring/                    ← Prometheus + MMD drift
│   └── utils/
├── scripts/
│   ├── export_onnx.py                 ← MANDATORY for production
│   ├── validate_onnx.py               ← Numerical parity check
│   └── save_drift_baseline.py
├── tests/                             ← Comprehensive test suite
├── requirements.txt
└── requirements.inference.txt         ← Minimal prod deps
```

---

## Requirements

### Hardware
- **Training**: GPU(s) with ≥16 GB VRAM recommended (mixed precision supported)
- **Inference**: CPU sufficient; NVIDIA GPU for TensorRT acceleration
- **Memory**: ≥8 GB RAM (16+ GB recommended for large WSIs)

### Software
- Python 3.11
- Docker + Docker Compose (recommended for production)
- OpenSlide libraries (`libopenslide0`)

### Models (mandatory for production)
- Pre-trained **UNI** backbone (via Hugging Face `MahmoodLab/uni`)
- Exported ONNX models (run `scripts/export_onnx.py` after training)
- Genomic preprocessing pipeline (`models/genomic_pipeline.joblib`)

### Secrets (environment variables)
- `JWT_SECRET_KEY`
- `ENCRYPTION_KEY_V1` (and `ENCRYPTION_KEY_CURRENT=V1`)
- `HF_TOKEN` (for gated backbones)

See full **Deployment Checklist** in the repository root.

---

## Installation & Quick Start

```bash
git clone <repo-url>
cd cancerAI_v12

# 1. Training / dev environment
pip install -r requirements.txt

# 2. Production inference only
pip install -r requirements.inference.txt

# 3. Export models to ONNX (MANDATORY)
python scripts/export_onnx.py --model-dir models/ --output-dir models/

# 4. Validate numerical parity
python scripts/validate_onnx.py --model-dir models/

# 5. Run with Docker (recommended)
docker compose up -d
```

API will be available at `http://localhost:8000`

---

## Configuration

- Main config: `config/config.yaml` (Pydantic-validated)
- Clinical rules: `config/rules/rules_v1.yaml` (hot-reload via `POST /admin/rules/reload`)
- Environment variables for secrets and deployment flags

---

## API Usage

**Key endpoints** (all require Bearer token):

- `POST /predict/image` — WSI only (returns cancer prob + top-K tiles)
- `POST /predict/multimodal` — WSI + genomic JSON (full output + rules)
- `GET /health`
- `POST /admin/rules/reload` (admin role only)
- `POST /token` (disabled in OIDC mode)

All responses include:
- `request_id` for tracing
- Full clinical disclaimer
- `human_review_required: true`
- `deferred_to_human` flag when uncertainty is high

---

## Deployment

- **Lean inference image** (`Dockerfile.inference`) recommended for production
- Prometheus metrics exposed on port 9090
- Request-ID correlation middleware for full observability
- Non-root user, read-only volumes, tmpfs for security

See `docker-compose.yml` and the **Deployment Checklist** in the repo.

---

## Current Limitations

1. **Regulatory status**: Research scaffold only — no clearance for clinical use.
2. **Data requirements**: Trained on specific cohorts (external validation required for new sites).
3. **WSI support**: Limited to OpenSlide-compatible formats (TIFF, SVS, etc.). No native DICOM WSI streaming.
4. **Genomic input**: Fixed 128-feature schema; no support for raw VCF/FASTQ.
5. **ONNX MC Dropout**: Approximated via input tiling (memory-intensive at very high `mc_samples`).
6. **Single-cancer focus**: Current models are general but were validated on specific tumour types (not multi-cancer out-of-the-box).
7. **No PACS/LIS integration**: Standalone API only.
8. **Explainability scope**: Tile-level attention is provided, but full saliency maps or whole-slide overlays are not generated by default.

---

## Future Upgrades / Roadmap

### Near-term (v12.x)
- Native support for additional backbones (CONCH, Virchow, H-optimus)
- Full attention heatmap overlay endpoint (PNG/SVG)
- Automated baseline generation scripts for drift detectors
- Improved ONNX MC Dropout (export-time stochastic nodes)
- Expanded rules DSL with temporal conditions and patient history

### Medium-term (v13)
- Multi-modal fusion with radiology (CT/MRI) and IHC images
- Federated learning support
- Prospective clinical trial pipeline + ISO 14971 risk management templates
- PACS integration via DICOMweb
- Model versioning + A/B testing endpoint

### Long-term
- Full regulatory pathway support (FDA 510(k) / CE Class IIa documentation templates)
- Real-time slide streaming + incremental inference
- Patient-specific outcome prediction (survival, recurrence)
- Integration with electronic health records (EHR) via FHIR

---

## License

This project is released under the **Research & Non-Commercial Use License** (see `LICENSE`). Commercial or clinical deployment requires separate licensing and regulatory approval.

---

**CancerAI v12** — Built as a transparent, auditable, and clinician-centric research platform.


*Last updated: April 2026*
*Developer: James Squire*
```

