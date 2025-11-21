---
language:
- en # ISO language tag
tags:
- project:genesis                  # include on all GENESIS project models
- project:bnl-ai                   # short model team name
- type:model                       # {agent, eval, framework, model, etc.}
- science:neuroscience             # e.g., materials, biology, lightsource, fusion, climate, etc.
- risk:general                     # {general, reviewed, restricted}
license: TBD                       # SPDX license identifier to be decided
base_model: TBD                    # base LLM / vision FMs to be decided
datasets:
  - https://abcdstudy.org/         # ABCD multimodal MRI / fMRI
  - https://www.ukbiobank.ac.uk/   # UK Biobank brain imaging
  - https://www.humanconnectome.org/ # HCP
  - https://healthybrainnetwork.org/ # HBN (sMRI/fMRI/EEG)
  - Multisite EEG / ECoG / sEEG / MEG datasets (TUEG, NSRR, etc.; details to be done)
  - Large-scale iEEG–video datasets (URLs and catalog to be done)
  - Text and video corpora aligned with neuroimaging / electrophysiology (details to be done)
metrics:
  - Training / validation loss for self-supervised and contrastive objectives
  - Task metrics (accuracy, AUROC, AUPRC, R², MAE; task list to be finalized)
  - Cross-modal retrieval / alignment metrics (definitions to be done)
  - Calibration and uncertainty metrics (ECE, Brier score; protocols to be done)
---

# NeuroX-Fusion: Unified Foundation Model of Brain for Transformative Neuroscience

A planned **≈130B-parameter multimodal foundation model** that unifies **MRI**, **electrophysiology**, **video**, and **language** into a single shared representation of brain structure and function, enabling cross-modal decoding, in-silico perturbations, and population-scale predictive neuroscience.

*Last Updated*: 2025-11-21  

> **Status:** Design / proposal stage. Model architecture and training plan are specified; large-scale implementation and public release are **to be done**.

---

## Developed by

- **Brookhaven National Laboratory (BNL)** – Artificial Intelligence Department  

**Principal Investigator & Point of Contact (POC)**  
- **David Park** – Brookhaven National Laboratory, AI Department  
  - Email: dpark1@bnl.gov  

**Co-PI**  
- **Shinjae Yoo** – Brookhaven National Laboratory, AI Department  

(Additional collaborators and institutions will be added in future versions of this model card.)

---

## Contributed by

- Brookhaven National Laboratory – AI Department  

(External contributors will be listed once model code and checkpoints are publicly released.)

---

## Model Changelog

- **2025-11-21** – Initial design-stage model card (no public weights; implementation in progress).

---

## Model short description

**NeuroX-Fusion** is a planned **unified brain foundation model** that combines:

- **NeuroX-MRI**: a ≈91B parameter MRI-centric foundation model (sMRI, dMRI, fMRI).  
- **NeuroX-Ephys**: a ≈50B parameter electrophysiology foundation model (EEG, ECoG, sEEG, MEG), inspired by the **DIVER** EEG foundation model.  
- A shared **Mixture-of-Experts (MoE) multimodal LLM** that fuses MRI and EPhys with video and language.

The model is designed to support **cross-modal decoding** (e.g., brain → text, brain → video features) and **in-silico experiments** (e.g., virtual lesions, stimulation).

---

## Model description

> All concrete implementation details (exact code, configs, and training runs) are **to be done**. This section describes the planned architecture and training strategy.

### High-level architecture

NeuroX-Fusion is structured as a three-part stack:

1. **NeuroX-MRI (~91B parameters, planned)**  
   - **4D fMRI encoder**:  
     - Swin-based 4D fMRI Transformer (SwiFT-style) for volumetric spatiotemporal data (3D space + time).  
   - **sMRI / dMRI encoders**:  
     - 3D ViT-style encoders with Mixture-of-Experts (MoE) layers for structural MRI and diffusion MRI.  
   - **Multimodal MRI alignment**:  
     - Contrastive self-supervised training to align sMRI–fMRI and sMRI–dMRI, using sMRI as a structural anchor.  
   - **Language alignment**:  
     - Lightweight adapters that project MRI representations into the latent space of a large language model (LLM), linking brain data to natural language descriptions.

2. **NeuroX-Ephys (~50B parameters, planned)**  
   - **DIVER-style electrophysiology backbone** (EEG foundation model reference: https://github.com/dyhan316/DIVER):  
     - Full spatiotemporal self-attention over channels and time.  
     - Channel-permutation equivariance for arbitrary electrode layouts.  
     - Rotary positional embeddings in time and Sliding Temporal Conditional Positional Encoding (STCPE).  
   - **Multimodal EPhys**:  
     - Extends EEG architecture to **EEG, ECoG, sEEG, MEG**, with modality-aware embeddings and MoE layers.  
   - **Language & vision grounding**:  
     - EPhys ↔ text contrastive/instruction tuning.  
     - EPhys ↔ video alignment, plus video ↔ LLM alignment, so brain activity can be interpreted via both text and video.

3. **NeuroX-Fusion (unified ≈130B multimodal MoE LLM, planned)**  
   - **Multimodal LLM core**:  
     - A large MoE LLM that accepts MRI and EPhys embeddings (and optionally video embeddings) via cross-modal adapters.  
   - **Language as a “semantic hub”**:  
     - All modalities are projected into a shared language-centric latent space, enabling cross-modal reasoning even when MRI and EPhys are not simultaneously acquired.  
   - **Training stages (to be implemented)**:  
     1. Pretrain MRI and EPhys encoders separately at scale.  
     2. Train modality-specific experts and adapters in the LLM.  
     3. Perform parameter-efficient fine-tuning (e.g., QLoRA) of the LLM core for fusion tasks.

### Scientific objectives

Planned objectives:

1. **Decode internal states (“decode the unspoken mind”)**  
   - Continuous decoding of affective and cognitive states, and discovery of health-relevant signatures, from multimodal brain data.

2. **Bridge scales and modalities**  
   - Jointly model:  
     - Structural anatomy (sMRI),  
     - White-matter connectivity (dMRI),  
     - Mesoscale dynamics (fMRI),  
     - Millisecond-scale electrophysiology (EEG / ECoG / sEEG / MEG).  

3. **Enable in-silico experiments / digital twins**  
   - Use the model as a “digital twin” for the brain to simulate lesions, stimulation, or cognitive manipulations and predict multimodal consequences.

4. **Advance exascale AI for science**  
   - Develop scalable training recipes and data pipelines that are reusable across scientific domains.

---

## Finetuned from model (optional)

- NeuroX-Fusion will be **built on top of one or more large open-source LLMs and vision foundation models**, but the exact base checkpoints are **to be decided**.  
- This section will be updated once the base LLM and vision models are formally selected and documented.

---

## Model Type

- **Multimodal Mixture-of-Experts Transformer**  
- MRI encoders:  
  - 4D Swin-based fMRI Transformer (SwiFT-style)  
  - 3D ViT/Transformer for sMRI and dMRI with MoE layers  
- Electrophysiology encoder:  
  - DIVER-style, fully channel-permutation-equivariant transformer for EEG / ECoG / sEEG / MEG  
- Fusion core:  
  - Large MoE LLM with cross-modal adapters for MRI, EPhys, video, and text  

---

## Inputs and outputs

### Inputs

**MRI stream (NeuroX-MRI)**

- **sMRI**: T1/T2-weighted structural MRI volumes.  
- **dMRI**: Diffusion MRI volumes, tractography-derived connectivity features.  
- **fMRI**: Resting-state and task-based fMRI (4D spatiotemporal volumes).  
- Optional: demographic and clinical covariates (age, sex, clinical scores, behavioral measures).

**EPhys stream (NeuroX-Ephys)**

- **EEG**: Multi-channel scalp EEG with heterogeneous montages and sampling rates.  
- **ECoG / sEEG**: Intracranial electrophysiology, including grid and depth electrodes.  
- **MEG**: Whole-head magnetoencephalography.  
- Per-channel metadata: sensor coordinates, modality labels, sampling rates.

**Auxiliary modalities**

- **Video**: Naturalistic and task-related behavioral video synchronized with EPhys.  
- **Text**: Task instructions, clinical notes, behavioral descriptions, and general-domain text corpora for LLM pretraining.

### Outputs

All outputs described here are **planned**, and concrete heads/APIs are **to be implemented**.

- **Latent embeddings**  
  - Shared, modality-agnostic representations of brain state for each input (MRI/EPhys/video/text).  

- **Decoding tasks**  
  - Cognitive and affective state decoding (e.g., task condition, attention, arousal).  
  - Regression of clinical or behavioral scores (for research use only).  
  - Quality control or modality classification tasks.  

- **Cross-modal mapping**  
  - MRI → EPhys feature prediction, EPhys → fMRI network embedding.  
  - Brain → text (narrative or structured descriptions).  
  - Text → brain (predicted pattern prototypes for hypothetical scenarios).  

- **Generative and in-silico capabilities**  
  - Virtual structural lesions or stimulations and corresponding predicted changes in latent representations or outputs.  
  - Brain-guided multimodal generation (e.g., brain → text / audiovisual features).

---

## Compute Infrastructure

> The following infrastructure plan is based on proposal documents; exact usage is **to be executed and documented**.

### Hardware (planned)

- **System**: Aurora (ALCF) or comparable leadership-class system.  
- **Compute request**:  
  - ≈1.3 million node-hours across 3 years for pretraining and fusion.  
- **Parallelism**:  
  - Up to 256–512 nodes for large Swin/EPhys/LLM-MoE training runs.  
  - Smaller node counts for ablations, scaling studies, and evaluation.

### Storage (planned)

Approximate scratch requirements during peak training:

- ≈300 TB multimodal MRI  
- ≈150 TB electrophysiology  
- ≈120 TB aligned video  
- ≈150 TB for checkpoints, logs, and intermediate artifacts  

Total: **≈600 TB** projected high-water mark.

### Software stack (planned; formal implementation to be done)

- **Core training**: PyTorch + DeepSpeed (ZeRO, pipeline/model parallelism, MoE).  
- **Medical imaging**: MONAI, NiBabel-based pipelines.  
- **EPhys**: MNE-Python and custom scalable preprocessing / artifact removal pipelines.  
- **Optimization**:  
  - Mixed precision (BF16/FP16)  
  - Flash-style efficient attention kernels  
  - µTransfer or similar techniques for hyperparameter transfer across model sizes  

Exact environment specifications (conda, spack, containers) are **to be published** with the first released code.

---

## Papers and Scientific Outputs

### Foundational / directly related internal work

- **DIVER** (EEG foundation model; GitHub: https://github.com/dyhan316/DIVER)  
  - Provides the architectural basis for NeuroX-Ephys: full spatiotemporal attention, channel-permutation equivariance, and STCPE for arbitrary EEG montages.  

- **NeuroX-Fusion proposals (ASCR / INCITE / internal)**  
  - Define the large-scale design, data plan, and exascale compute strategy for NeuroX-MRI, NeuroX-Ephys, and unified NeuroX-Fusion.

(Full bibliographic references will be added when internal manuscripts are public.)

### Related work

These works are closely related to NeuroX-Fusion’s goals in large-scale fMRI/EPhys modeling, cross-modal decoding, causal/spatiotemporal inference, and semantic evaluation:

- **SwiFT: Towards Large-Scale 4D fMRI Foundation Models**  
  - **Venue**: NeurIPS 2023  
  - Swin-based 4D Transformer for fMRI foundation modeling at scale.  
  - URL: https://arxiv.org/abs/2307.05916  

- **SwiFT v2: Towards Large-scale Foundation Model for Human Brain**  
  - **Venue**: CCNeuro 2025 (abstract)  
  - Extends SwiFT with improved scaling and generalization for fMRI foundation models.  
  - URL: https://2025.ccneuro.org/abstract_pdf/Choi_2025_SwiFT_V2_Towards_Large-scale_Foundation_Model.pdf  

- **SCENT**  
  - **Venue**: ICML 2025  
  - Transformer-based semantic brain decoding framework, conceptually aligned with NeuroX-Fusion’s brain→text / semantic alignment goals.  
  - URL: https://openreview.net/forum?id=vhjuemZuRU  

- **OMNIFIELD**  
  - Universal neural field for brain imaging representation learning (large-scale, modality-flexible representation).  
  - URL: https://www.arxiv.org/abs/2511.02205  

- **STACI: Spatio-Temporal Aleatoric Conformal Inference**  
  - **Venue**: NeurIPS 2025  
  - Provides methods for calibrated, spatiotemporal uncertainty quantification, relevant to future UQ for NeuroX-Fusion outputs.  

- **Spatiotemporal Learning of Brain Dynamics from fMRI Using Frequency-Specific Multi-Band Attention for Cognitive and Psychiatric Applications**  
  - **Venue**: arXiv preprint (arXiv:2503.23394)  
  - fMRI model with frequency-specific multi-band attention for cognitive and psychiatric prediction tasks.  

- **SEED: Towards More Accurate Semantic Evaluation for Visual Brain Decoding**  
  - **Venue**: arXiv preprint (arXiv:2503.06437)  
  - Proposes improved semantic evaluation metrics for visual brain decoding, relevant to how NeuroX-Fusion’s decoding quality is assessed.  

- **GST-UNet: Spatiotemporal Causal Inference with Time-Varying Confounders**  
  - **Venue**: NeurIPS 2025  
  - Spatiotemporal causal inference architecture that informs how to reason about interventions and confounding in brain data.  

- **Macro2Micro: Cross-modal Magnetic Resonance Imaging Synthesis Leveraging Multi-scale Brain Structures**  
  - Cross-modal MRI synthesis across scales, conceptually related to NeuroX-Fusion’s structural–functional alignment and modality completion.  

- **Revisiting Your Memory: Reconstruction of Affect-Contextualized Memory via EEG-guided Audiovisual Generation**  
  - **Venue**: ACM Multimedia 2025, CogMAEC Workshop (Oral)  
  - EEG-guided audiovisual generation of affect-contextualized memory traces, closely aligned with NeuroX-Fusion’s brain→multimodal generation ambitions.  

(Additional related work will be added as the project matures.)

---

## Model License

- **License**: TBD  
- Direction: an open-source, permissive license (e.g., Apache-2.0 or BSD-3-Clause), subject to data-use restrictions and DOE program requirements.  
- Exact SPDX identifier and license text will be finalized when code and weights are ready for release.

---

## Contact Info and Model Card Authors

- **PI & POC**  
  - **David Park**  
    - Brookhaven National Laboratory, AI Department  
    - Email: dpark1@bnl.gov  

- **Co-PI**  
  - **Shinjae Yoo**  
    - Brookhaven National Laboratory, AI Department  

- **Model card authors**  
  - BNL AI Department – NeuroX-Fusion project team  

---

# Intended Uses

## Intended Use

NeuroX-Fusion is intended as a **research-only** multimodal neuroscience foundation model, for:

- **Multimodal representation learning**  
  - Learning common latent spaces for sMRI, dMRI, fMRI, EEG/ECoG/sEEG/MEG, video, and text.  

- **Brain state decoding**  
  - Decoding cognitive and affective states, and learning research biomarkers for neurological/psychiatric conditions.  

- **Cross-modal mapping / completion**  
  - Predicting EPhys signatures from MRI or vice versa, harmonizing across cohorts and acquisition sites.  

- **In-silico experimentation**  
  - Simulating hypothetical perturbations (lesions, stimulation, task changes) to generate hypotheses about brain dynamics.  

- **AI-for-science methodology**  
  - Studying large-scale multimodal MoE training, data-centric pipelines, and exascale infrastructure.

### Primary Intended Users

- Neuroscience and psychiatry researchers  
- Clinician-scientists (for research, not clinical decision support)  
- AI/ML researchers working on multimodal and scientific foundation models  
- DOE and partner institutions exploring AI-for-science

### Mission Relevance

- Addresses DOE AI grand challenges by building an exascale-scale foundation model for a complex, data-rich scientific domain.  
- Provides patterns and infrastructure that can be adapted to other domains (e.g., climate, materials, fusion).

---

## Out-of-Scope Use Cases

NeuroX-Fusion **must not** be used for:

- **Clinical diagnosis or treatment decisions**, or any regulated medical use, without extensive validation and regulatory approval.  
- **Real-time medical device control** as a stand-alone controller.  
- **Surveillance or covert mental-state inference** without fully informed consent.  
- Consumer claims of “mind reading” or thought decoding that exceed validated capabilities.  

All high-stakes applications must involve domain experts, IRB/ethics review, and appropriate regulation.

---

# How to use

> No public code, checkpoints, or APIs are available yet. This section describes planned usage; all concrete implementation is **to be done**.

## Install Instructions (to be done)

Planned:

- Python package (e.g., `neurox_fusion`) published to PyPI/conda.  
- Reference Docker/Singularity images with dependencies (PyTorch, DeepSpeed, MONAI, MNE, etc.) pre-installed.  
- Example Slurm/PBS scripts for multi-node training and inference on leadership-class systems and typical GPU clusters.

Exact commands and environment files will be documented with the initial public release.

## Training configuration (to be done)

High-level plan:

1. **NeuroX-MRI pretraining**  
   - Self-supervised and contrastive objectives over sMRI, dMRI, and fMRI.  
   - SwiFT-style 4D Transformer and 3D ViT with MoE layers.  

2. **NeuroX-Ephys pretraining**  
   - DIVER-style, channel-equivariant transformer for EEG/ECoG/sEEG/MEG.  
   - Self-supervised learning across multiple EPhys datasets.  

3. **Alignment with LLM**  
   - Train modality adapters and contrastive/instruction-tuned interfaces between MRI/EPhys and the LLM.  

4. **Unified NeuroX-Fusion training**  
   - Train modality-specific MoE experts within the LLM.  
   - Use parameter-efficient fine-tuning to adapt the LLM core to multimodal tasks.

Concrete hyperparameters (learning rates, batch sizes, optimizer configs, training schedules) will be defined in open configuration files.

## Inference configuration (to be done)

Planned capabilities:

- Encode MRI, EPhys, video, and text into shared latent spaces.  
- Run task-specific decoding heads (classification/regression).  
- Perform cross-modal retrieval and generation (e.g., brain→text, text→brain pattern prototypes).  

APIs will be designed to support both leadership-class systems and standard multi-GPU servers.

---

# Code snippets of how to use the model

> **Note:** The following is illustrative pseudocode. Actual APIs will be defined later.

```python
# PSEUDOCODE ONLY – real API will differ

from neurox_fusion import NeuroXFusionModel, NeuroXConfig

cfg = NeuroXConfig.from_pretrained("bnl-ai/neurox-fusion-130b")  # to be done
model = NeuroXFusionModel.from_pretrained(cfg)
model.eval()

batch = {
    "smri": smri_volumes,        # [B, C, X, Y, Z]
    "dmri": dmri_volumes,        # [B, C, X, Y, Z]
    "fmri": fmri_4d,             # [B, T, X, Y, Z]
    "ephys": eeg_timeseries,     # [B, C, T]
    "video": video_clips,        # optional
    "text": text_tokens,         # tokenized prompts
}

with torch.no_grad():
    outputs = model(batch, task="brain_to_text")  # or "decode_state", "cross_modal_map", etc.

print(outputs["text"])
