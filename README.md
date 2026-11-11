# Synthagent

Synthagent generates synthetic records for small tabular datasets with a large language model (LLM). A deterministic control layer checks and repairs every LLM response and requests the missing records, so the requested dataset is delivered with the columns and types of the real data, together with an audit trail of every request.

This repository accompanies the paper *Synthagent: Reliable LLM-Based Synthetic Data Generation for Small Healthcare Datasets* (ICDE 2027 submission).

- `src/Synthagent/` - implementation, prompts, experiment runner, datasets and results.
- `src/Synthagent/scripts/reproduce/` - how to reproduce every experiment (`README.md`, `run_all.sh`).
- `Synthagent__An_Agentic_Framework_for_Synthetic_Data_Generation/` - paper source and the scripts that regenerate every table and figure from the results.
