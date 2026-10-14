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

## Confirmation (to be completed before any confirmation run)

The frozen configuration, the confirmation dataset and the predictions are written in the section below once
development ends, and committed before the confirmation data are downloaded.
