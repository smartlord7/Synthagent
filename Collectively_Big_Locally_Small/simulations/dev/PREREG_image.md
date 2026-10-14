# GHB for images: development on ISIC 2019, confirmation pre-registered

Written 2026-10-13, before any run of `isic_image.py`.

## Status of ISIC 2019

ISIC 2019 was the pre-registered test of `PREREG_flamby.md` (prediction 4, which held with every borrowing method
at the local level). From here on it is **development data** for the image version of GHB: every design choice
below may be changed after looking at ISIC 2019 results. No ISIC 2019 number obtained under this file is a
confirmation, and the paper reports it as development.

## What is being developed (`ghb_image.py`)

1. A public projection: PCA of the frozen encoder's features of the public ISIC 2019 Test images (unlabeled, not
   part of FLamby's six centers), replacing the 16-dimensional random projection.
2. One softmax head over the 8 classes, one budget, instead of eight one-vs-rest heads with an eighth each.
   Sensitivity of a score sum under replacement: 2 sqrt(2) G, used by GHB and every literature method.
3. GHB's curvature from the public images (the softmax information needs features and the model, not labels), so
   the only private release is the secure sum of scores (R1 Newton rounds).
4. Logit adjustment for class imbalance and label shift, applied to every method alike: each hospital predicts
   argmax_k s_k - log pi_k with pi the class mix the model implies on its own training images.

Variants explored: R1 in {1, 3}, free intercepts (class mix learned locally), PCA dimension d.

## Confirmation on Camelyon17 (frozen 2026-10-14, committed before any Camelyon17 image or label is downloaded)

Development on ISIC 2019 ends here. Its result (balanced accuracy, 10 repetitions, pFedMe with corrected noise): GHB
about 0.58 at eps = 10 against about 0.53 for local estimation and the best tuned literature methods; at eps = 1 GHB is
level with local. Nothing below is changed after Camelyon17 data are seen; every prediction is reported as held or not
held, whatever the result.

### Data source (a declared deviation from "Camelyon17-WILDS as downloaded by the WILDS package")

Camelyon17-WILDS (Koh et al., 2021; Bandi et al., 2019) is distributed only from worksheets.codalab.org, whose TLS
certificate expired on 2026-10-07 (checked 2026-10-14: the host answers, the certificate is refused; verification is not
disabled). Third-party copies (Hugging Face, Kaggle, Zenodo) are blocked by this machine's network policy. The patches
are therefore **rebuilt from the official CAMELYON17 release** (AWS Open Data, `https://camelyon-dataset.s3.amazonaws.com/CAMELYON17/`,
the organizers' own copy, with MD5 checksums) **with the WILDS preprocessing code** (`p-lambda/wilds`,
`dataset_preprocessing/camelyon17/`, commit 472677590de351857197a9bf24958838c39c272b), reimplemented in
`simulations/camelyon_image.py` with one change of library: slides are read by HTTP range requests with `tifffile`
(`simulations/camelyon_slides.py`, tested on a CAMELYON16 slide only) instead of OpenSlide from a local copy (the 50
annotated slides are about 150 GB, more than this machine's disk). The recipe, unchanged:

- the 50 slides with lesion annotations (10 per center; center = patient // 20, patients 0 to 99), processed in sorted
  file-name order; annotation groups "Tumor" and "Exclusion" of the official XML are WILDS's "metastases" and
  "normal" (same polygons; the official README states the correspondence); the XML files are checked against
  `checksums.md5`;
- PATCH_LEVEL = 2, MASK_LEVEL = 4, CENTER_SIZE = 32 (patches 96 x 96 at level 2 on a 32-pixel grid), margin 5 grid
  cells; tissue mask = Otsu threshold of the HSV saturation of the level-4 image (WILDS's own BGR2HSV call on RGB
  input, kept as is); a cell is tumor if any of its central region is in a tumor polygon (threshold 0), else normal if
  more than 20% of it is non-tumor tissue;
- class balancing of `generate_final_metadata.py` (np.random.seed(0): all tumor cells of a slide unless it has fewer
  normal than tumor cells, then per center as many normal as tumor cells). The WILDS train/val split is ignored.

The result is the Camelyon17-WILDS recipe on the same 50 slides, not bit-identical to the released metadata (row order
and the random draws follow from our file order). Status of the data seen so far: while checking reachability, the
first five lines of `CAMELYON17/stages.csv` (slide-level labels of four slides of patient 0) and the first 1.5 KB of one
annotation file (`patient_004_node_4.xml`) were printed. No image, patch, feature or result of Camelyon17 has been
obtained.

### Hospitals, size and split

- Five hospitals = the five CAMELYON17 centers; binary task, tumor vs normal patch; K = 2 classes, softmax head.
- **Pool**: from each center's balanced metadata, 600 tumor and 600 normal patches drawn with
  `np.random.default_rng(2028)` (centers in order 0 to 4): 1,200 per hospital, 6,000 in all. Only these are extracted.
- **Split** (per repetition r = 0, ..., 9): stratified random split within each hospital with
  `np.random.default_rng([2028, r])`, hospitals in order 0 to 4, classes in order 0, 1: 100 patches of each class to
  training (n = 200 per hospital), the other 1,000 (500 of each class) to test.
- **Why 200**: the paper's setting is a network of small hospitals. With d = 32 public directions and an intercept the
  softmax head has 66 coefficients, so 200 training patches make local estimation noisy and leave room for borrowing,
  while each hospital still has enough images to train on its own (a site that could label 200 patches). The full
  centers (tens of thousands of patches each) would put every method at its ceiling and the question would be moot.
  The 1,000 test patches per hospital keep the evaluation noise small; they are never used for training, tuning or the
  projection.
- Privacy unit: one patch (as one image in ISIC 2019). Patches of the same slide, and of the same patient, are
  correlated, so patch-level differential privacy is weaker than patient-level privacy; this is stated with the
  results. Train and test patches of a hospital come from the same slides (the pre-registered random split), which
  makes the task easier for every method alike.

### Public images for the projection and GHB's curvature

10 slides of the CAMELYON17 **test set** (patients 100 to 199, never annotated in the release and disjoint in patients
from the 50 hospital slides): 10 distinct patients drawn with `np.random.default_rng(2029)`, one node each drawn with the
same generator; on each slide, the WILDS tissue mask and grid as above, the cells with more than 20% tissue, and 200 of
them drawn uniformly (same generator): 2,000 unlabeled public patches. They are public (released by the organizers),
unlabeled (no label of them is read or used), and not part of any hospital's training or test patches; they play the
role of the public ISIC 2019 Test images in development. Known risk, accepted in advance: most test-set slides are
tumor-free, so the public patches are mostly normal tissue (the curvature needs features and the model, not labels;
GHB's first Newton step is computed at a start where every public patch gets the same probability).

### Features and model (as `isic_image.py`)

- Encoder: FLamby's EfficientNet-B0 (`efficientnet_pytorch`), ImageNet weights `efficientnet-b0-355c32eb.pth` from
  the GitHub release (sha256 prefix 355c32eb, checked), frozen, eval mode; input: the 96 x 96 RGB patch resized to
  224 x 224 (bilinear), scaled to [0, 1], ImageNet mean/std normalization (no color constancy, which was ISIC's
  dermoscopy-specific step); output: the 1,280-dimensional global average pool of `extract_features`. CPU PyTorch.
- Projection: PCA with d = 32 components fitted on the 2,000 public patches (`random_state=0`); per-hospital
  standardization with the hospital's training statistics, /3, clip to [-1, 1], intercept (P = 33, G = sqrt(33));
  public rows standardized with their own statistics.
- Methods (repetition r uses seed [2027, r]; eps in {1, 10, inf}, delta = 1e-5):
  - Local (prior sd 2.5), Local with evidence prior, Pooled (non-private reference);
  - **GHB = `isic_image.VARIANTS["GHB R1=1"]`**: R1 = 1, init_share = 0.05, damp 1, no free intercept (the intercept
    is handled by the logit adjustment), borrowing along all 32 PCA directions, safety components on;
  - the literature: the six families of `real_v3.GRIDS` (DP-FedAvg + FT (McMahan et al., 2018), DP-FedAvg-SA + FT,
    APFL (Deng et al., 2020), Ditto (Li et al., 2021), pFedMe (T. Dinh et al., 2020) with the corrected noise of
    `ghb_image.pfedme`, Clustered DP-FedAvg (Ghosh et al., 2020)) in their multinomial versions
    (`ghb_image.baselines`, sensitivity 2 sqrt(2) G), at every grid configuration. **Realistic arm**: the configuration
    of `dev/v4_realistic_configs.json` (the eps = 10 one at eps = inf). **Oracle arm**: per family and budget, the
    configuration with the best repetition-averaged mean (on the test patches; an upper bound no real network could
    reach).
- Prediction rule for every method: logit adjustment by the class mix the model implies on the hospital's own training
  images, restricted to the classes it has seen (`adjust="model"`, `support`), as in development.

### Outcomes

- Primary: balanced accuracy per hospital on its test patches, mean over the five hospitals, mean over 10 repetitions.
- Secondary: log-loss per hospital (mean over its test patches of -log p(y) with p the logit-adjusted softmax over
  the seen classes, clipped at 1e-15), mean over hospitals and repetitions.
- Comparisons are of unrounded means; the table rounds to three decimals.

### Predictions

1. At eps = 10, GHB's mean balanced accuracy is above Local's.
2. At eps = 10, GHB's mean balanced accuracy is above that of every literature method in the realistic arm.
3. At eps = 1, GHB's mean balanced accuracy is not more than 0.005 below Local's (level with local, as in development).
4. At eps = inf, GHB's mean balanced accuracy is above both Local's and Pooled's.
5. At eps = 10, GHB's mean log-loss is below Local's.

Each prediction will be reported as held or not held. If a slide cannot be read or a center yields fewer than 600
patches of a class, that is reported and the pool uses what exists; no other change.
