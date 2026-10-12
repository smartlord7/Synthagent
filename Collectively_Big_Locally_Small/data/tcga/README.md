# TCGA clinical data by tissue source site

Open-access clinical data of six TCGA projects (LUSC, HNSC, BLCA, STAD, LIHC, KIRC), downloaded from the NCI Genomic
Data Commons API (https://api.gdc.cancer.gov/cases, data release 46.0) with the fields `submitter_id`,
`demographic.{vital_status, days_to_death, age_at_index, sex_at_birth}` and
`diagnoses.{days_to_last_follow_up, age_at_diagnosis, ajcc_pathologic_stage, diagnosis_is_primary_disease}`.
The tissue source site (the contributing institution) is the second field of the patient barcode, `TCGA-XX-YYYY`.
Open-access TCGA clinical data are available without restriction under the NIH GDC data access policies. Used by
`simulations/real_networks.py`.
