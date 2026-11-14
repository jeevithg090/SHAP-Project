# `data/` — local inputs (not version-controlled)

Everything in this folder except this README is git-ignored. The source workbook
holds **identifiable patient information** and must never be committed.

Expected files (override the locations with the variables in `.env.example`):

| File | Variable | Contents |
|---|---|---|
| `Version 2- MIT calculation table.xlsx` | `HAND_INJURY_WORKBOOK_PATH` | One sheet, one row per patient. A human-readable form block is followed by a machine-readable block that starts at the `patient_id` column, with left/right (`_l` / `_r`) injury findings and pre-computed scores (MHISS, Atthakomol, MUES, Strickland, IPS). |
| `Data_Dictionary_v3.xlsx` | `HAND_INJURY_DATA_DICTIONARY_PATH` | Sheet `Data Dictionary` with at least `Column Name` and `Layer` (`input` / `target`); `Data Type` and `Allowed Values / Range` are used when present. |

Only the machine-readable block is loaded. Three answers from the form block
(`Side Involved`, `Injured Hand`, `Dominance`) are copied in to cross-check the
injured side, and `patient_id` is replaced by a pseudonym before anything is written
to `outputs/`.
