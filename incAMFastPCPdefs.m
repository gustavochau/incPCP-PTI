
% Ground-truth
% -----------------------------

INCPCP_GT_NONE       = 0;  % No GT is given
INCPCP_GT_MATFILES   = 1;  % GT is given as a 'golden' sparse component in mat-files.
INCPCP_GT_SPARSEMAT  = 2;  % GT is given as a 'golden' sparse component in a matrix.
INCPCP_GT_BINMASK    = 3;  % GT is given as a binary mask in jpg/png files.
INCPCP_GT_BINMASK_MATFILES    = 4;  % GT is given as a binary mask in jpg/png files and also as a 'golden' sparse component in mat-files.
INCPCP_GT_BINMASK_SPARSEMAT   = 5;  % GT is given as a binary mask in jpg/png files and also as a 'golden' sparse component in a matrix.

INCPCP_COMPUTE_BINMASK_NONE   = 0;      % Do not compute binMask
INCPCP_COMPUTE_BINMASK_LM     = 1;      % Compute binMask via Leung-Malik (LM) Filter Bank + unimodal
INCPCP_COMPUTE_BINMASK_1MOD   = 2;      % Compute binMask via unimodal segmentation
INCPCP_COMPUTE_BINMASK_THRESH = 3;      % Compute binMask via a given (fixed) threshold

% TI Refinement
% -----------------------------

INCPCP_TI_REFINEMENT_STANDARD = 1;
INCPCP_TI_REFINEMENT_SEARCH   = 2;
INCPCP_TI_REFINEMENT_CSR      = 3;

INCPCP_TI_ROTATION_MOSAICPAD  = 4;
INCPCP_TI_ROTATION_ZEROPAD    = 5;

