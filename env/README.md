# Environments

- `../requirements.txt`: tested numerical stack for the portable development launcher, smoke test and score replay. Python 3.9.23 was recorded for the original runs; Python 3.9.6 was used in the fresh local test.
- `development.yml`: conda recipe pinning Python 3.9.23 and the recorded direct dependencies. Conda solving itself was not tested locally.
- `figures-python.txt`: adds pandas 2.3.2; installed and used for the eight figure-data stages.
- `early-cpu.txt`: recorded earlier numerical versions plus the preparation reader dependency. Untested on this release; not a full lockfile and not the original Docker image.
- `figures-packages.tsv`: recorded R packages. No claim that a complete resolved R environment is bundled.
- The legacy scGPT embedding used a separate GPU environment: Python 3.10, PyTorch 2.3.0 with CUDA 12.1, scGPT 0.2.4, and the whole-human checkpoint with hashes in `metadata/RAW_SOURCES.tsv`. This environment is not reconstructed here.

The original early Docker image used Python 3.11.2, NumPy 1.26.4, SciPy 1.16.3, scikit-learn 1.7.2 and PyTorch 2.2.0 with CUDA 12.1. Its recorded local image ID is `sha256:0a450c02d469c8bad7d43c248fee99789e2f408472b0e5a2d1e118e04c250b9a`; it is not a pullable public reference. The root Dockerfile is a new CPU environment for the later development analyses.
