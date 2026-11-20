# pipeline/

Production pipeline for Vāgdhenu:

- **Data prep:** cut → pair → train
- **Build / assemble / QC:** the steps used for the MBTN and Śrīmad Bhāgavatam deliverables

**Status: not yet released.** The scripts for this stage have not been published in this repository yet.

Related code that is already in the repo:
- `training/`: corpus download, alignment, IndicF5 fine-tuning and BigVGAN vocoder training
- `src/build_clips.py` and `src/extract_prosody.py`: clip building and prosody extraction
- `src/render.py` and `src/render_mlx.py`: batch rendering from shard JSON
