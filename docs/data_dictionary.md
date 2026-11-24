> **Working notes: the checked numbers are in docs/report.md**

# Data dictionary (Phase 1, inspection only)

Source: OSF dmhrw, Pletzer & Noachtar (2023). All numbers below come from `src/inspect_data.py`, `inspect2.py`, `inspect3.py`, `inspect4.py` (run with `.venv/bin/python -I`). No codebook was provided; the only documentation is the authors' R script (`data/Script_MC_RV20_EmotionRecognition.R`), used for variable coding only. Items marked **(inferred)** are my reading of the data, not stated by the authors.

## Files
| File | Rows x cols | One row = | Women |
|---|---|---|---|
| `RV20_Questionnaires_MC_20230519.sav` | 333 x 176 | one **session** of one woman (wide format) | 184 |
| `RV20_Mood_MC_20230519.sav` | 668 x 152 | one session x `Valence` (0/1) -> every session appears twice | 185 |
| `RV20_EmotionRecognition_MC_20230519.sav` | 19,980 x 173 | one **trial** (item) of the emotion-recognition task | 185 |

**Main modelling table = Questionnaires file.** It already holds hormones, phase, trait scores and all mood scores. Checked: the Mood file duplicates it (Valence 0 rows: `Affect` == `Affect_pos`, Valence 1: `Affect` == `Affect_neg`, DRSP matches exactly). It has 22 missing `Phase` values that the Questionnaires file does not, and 12 phase mismatches on matched sessions (to be resolved; I will use the Questionnaires file). The Emotion file is needed only for Phase 6a (accuracy per session).

## Keys and sample coding
- **ID:** `VPN` (participant). **Session:** `Session` 1-3. Unique key (VPN, Session) has no duplicates in the Questionnaires file.
- **Longitudinal vs cross-sectional:** no explicit flag column. The authors' script defines longitudinal as `VPN` in 401-499 and cross-sectional as `Session == 1`. Participants with 3 sessions have VPN 401-494. **(see question 1)**
- Sessions per woman (all 184): 109 with 1, 1 with 2, 74 with 3.
- **`include`** (1/0): the authors' analysis filter (all their scripts do `subset(include == 1)`). Included: 161 women, 272 rows (96 women with 1 session, 64 with 3, 1 with 2). Excluded: 61 rows (333 - 272). Reason for exclusion not documented in the files. `includeNC3all` is a second, stricter flag.
- `Version` (1-3): session-order/task-version counterbalancing **(inferred)**.

## Key variables (Questionnaires file)
| Column | Meaning | Missing (all rows) | Notes |
|---|---|---|---|
| `Phase` | cycle phase 0/1/2 | 0 | **Inferred** 0 = menses, 1 = peri-ovulatory, 2 = luteal (median cycle day 4 / 14 / 23; progesterone highest in 2). No value labels in file. |
| `cycle_day_forward` / `_backward` | days since menses onset / days to next menses (negative) | 0 / 11 | forward range 1-59, backward -41 to 8 |
| `estradiol` | salivary E2, pg/ml | 7 | median 0.74, max 3.35, skew 1.15; 13 exact zeros |
| `progesterone` | salivary P, pg/ml | 7 | median 38.9, max 652, skew 2.93; 9 exact zeros, 5 values > 300 |
| `PMS` | PSST total (= sum of PSST1-14, correlation 1.0) | 1 | range 0-22 |
| `BDI_RW`, `BAI_RW` | BDI, BAI raw sums | 1 each | 0-39, 0-40 |
| `Affect_pos`, `Affect_neg` | positive / negative affect (PANAS + extended PANAS items, mean) | 0 | 1-5 scale; `Affect_Diff` = pos - neg **(inferred)** |
| `PANAS_pos/neg`, `EPANAS_pos/neg` | the two sub-scales | 0 | |
| `DRSP` | DRSP total (14 items, sum) | 0 | 14-65 |
| `DRSP_psychisch` / `DRSP_physisch` | psychological (9 items) / physical (5 items) sum | 0 | `DRSP_psychisch + DRSP_physisch == DRSP` for all rows |
| `DRSP_physical` | **duplicate of `DRSP_psychisch` (identical in every row) - mislabelled; do not use** | 0 | |
| `STAI` | state anxiety, mean item score | 0 | 1-2.95 |
| `Age`, `IQ`, `education`, ... | demographics | few | `Sex` is constant (all female) |
| `HC1_type`, `PriorContraception` | prior hormonal contraception details (German free text) | blanks | 157/272 included rows have non-blank HC1_type; meaning (current vs prior use) unclear |

Emotion file: trial-level `Hit`, `RT`, `Emotion`, `Answer`, plus the same ID/phase/hormone columns repeated on every trial (19,980 trials in total, 185 women; trials per session not yet verified). Accuracy per session = sum of `Hit` (authors' script does this after removing outliers).

## Descriptives, included sample (n = 272 rows)
Hormones: E2 mean 0.88 (sd 0.57), P mean 72.4 (sd 77.8); no missing hormones among included rows. Included-sample mood/trait scores have no missing except `PMS` (1).
Rows per phase, all women (n=326 with hormones): phase 0: 115, phase 1: 95, phase 2: 116. Phase means (E2 / P, all rows): 0.64/32.5, 1.01/36.8, 0.93/127.2.

## Quality flags
1. Skewed hormones with exact zeros: `log` needs an offset (log(x+c)) or a transform that tolerates 0.
2. Repeated measures: 3 rows per longitudinal woman are not independent -> grouped CV mandatory.
3. Possible overlap between "cross-sectional" (Session == 1) and longitudinal women in the authors' definition (same woman could appear in both) -> leakage risk if used as separate samples.
4. Duplicate column `DRSP_physical`; Mood file phase gaps.
5. Hormone assays: salivary immunoassay validity limits (authors' caveat).

---
## Phase 1 addendum (decisions confirmed by the student; checks by `src/inspect5.py`, `src/design_checks.py`)

### Phase coding - confirmed
Authors' R script labels `Phase_factor` levels in order as `c("menses","pre-ovulatory","luteal")` (many plot calls), so **0 = menses, 1 = peri-ovulatory ("pre-ovulatory" in the script), 2 = luteal**. Agrees with my inference from cycle day and hormones.

### Decisions
- Filter: `include == 1` (272 rows, 161 women). Main phase source: **Questionnaires file**.
- **Longitudinal** = included women with >1 included session (65 women, 176 rows; all have VPN 401-494). **Cross-sectional** = women with exactly one included session (96 women, 96 rows). The samples are disjoint. The woman with 2 sessions is kept (she is longitudinal).
- Note: `include` works per session. Among the 65 longitudinal women, 46 keep all 3 sessions and 19 keep 2 (the other session was excluded by the authors' filter; 64 had 3 raw sessions + 1 had 2).

### Women / rows by sample and phase (after filter)
| Sample | Menses | Peri-ovulatory | Luteal | Rows | Women |
|---|---|---|---|---|---|
| Longitudinal | 65 | 51 | 60 | 176 | 65 |
| Cross-sectional (one session) | 39 | 18 | 39 | 96 | 96 |
| Total | 104 | 69 | 99 | 272 | 161 |

(Women per phase equals rows per phase within each sample because each woman has at most one session per phase.)

### Reconciliation with the paper (65 longitudinal, 153 cross-sectional)
- **Longitudinal: 65 women - matches.**
- **Cross-sectional: I get 96, the paper says 153. Not forced.** The authors' script defines cross-sectional as `Session == 1`, which gives 154 included rows = my 96 one-session women + 58 first sessions of longitudinal women. 154 is within 1 of the paper's 153 (a small difference I cannot explain from the files; possibly one further exclusion in a specific analysis). So the paper's "153" most likely **includes the first session of longitudinal women**, i.e. the samples overlap in the paper. My disjoint definition is deliberate (avoids leakage) and gives a smaller cross-sectional sample.

### Mood-file vs Questionnaires-file phase: the 12 mismatches (matched on VPN, Session)
11 are `Phase` missing in the Mood file (Questionnaires file has the value); 1 is a real disagreement. Cycle day is identical in both files. Questionnaires file is used.
| VPN | Session | Mood file | Questionnaires | cycle day | include |
|---|---|---|---|---|---|
| 626 | 1 | NaN | 2 | 17 | 0 |
| 633 | 1 | **1** | **2** | 12 | 1 |
| 641 | 1 | NaN | 1 | 15 | 0 |
| 645 | 1 | NaN | 2 | 24 | 0 |
| 652 | 1 | NaN | 2 | 21 | 0 |
| 659 | 1 | NaN | 1 | 13 | 0 |
| 668 | 1 | NaN | 1 | 16 | 0 |
| 676 | 1 | NaN | 1 | 11 | 0 |
| 685 | 1 | NaN | 1 | 15 | 0 |
| 689 | 1 | NaN | 1 | 9 | 0 |
| 694 | 1 | NaN | 1 | 15 | 0 |
| 698 | 1 | NaN | 2 | 24 | 0 |

Only VPN 633 is in the included sample (cycle day 12; the two files disagree on peri-ovulatory vs luteal). Also, the Mood file has a row (VPN 412, Session 1) that does not exist in the Questionnaires file.

### Update (Phase 8): the paper's "153 cross-sectional" is reconciled
In the 271-row analysis set (after dropping the one row with missing PSST, which was a first session), the authors' rule `Session == 1` selects **153 rows**, exactly the paper's 153 (computed in `src/build_report.py`; it equals my 95 one-session women plus the first sessions of 58 longitudinal women). The earlier mismatch of 1 (154 rows before the PSST drop) was therefore only that row. My disjoint cross-sectional sample (95 women) is a deliberate stricter definition, not an error in the data.
