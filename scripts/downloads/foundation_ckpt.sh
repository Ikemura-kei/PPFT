#! /bin/bash

# ===============================================================================
# this script downloads the CompletionFormer foundation model checkpoint, trained 
# on NYUDepth-V2 dataset, provided by the original authors, Yun et al.
# ===============================================================================

mkdir -p ckpts
cd ckpts && gdown https://drive.google.com/uc?id=1KJUZ4I-v9Nba0DDswHe2-Avq7yll---t