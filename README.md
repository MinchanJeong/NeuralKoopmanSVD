# Efficient Parametric SVD of Koopman Operator

[![NeurIPS2025](https://img.shields.io/badge/NeurIPS_2025-Poster-blue.svg)](https://neurips.cc/virtual/2025/loc/san-diego/poster/116395)
[![arXiv](https://img.shields.io/badge/arXiv-2507.07222-b31b1b.svg)](https://arxiv.org/abs/2507.07222)

**Official PyTorch Implementation** for the paper:  
> **Efficient Parametric SVD of Koopman Operator for Stochastic Dynamical Systems** > *Minchan Jeong\*, Jongha J. Ryu\*, Se-Young Yun, Gregory Wornell* > **NeurIPS 2025**

---

## Summary
- **Challenge:** Existing deep Koopman methods suffer from numerical instability and biased gradients.
- **Solution:** We propose to use LoRA-based optimization, which allows unbiased gradient estimates and requires no regularization.
- **Impact:** Scalable and stable training of dominant modes in high-dimensional systems (e.g., **Chignolin molecular simulations**), accurately recovering eigenfunctions.


## Code Release
The full source code and models will be released **by December 14, 2025**.  
We are currently finalizing the documentation and cleaning up the codebase for reproducibility.
