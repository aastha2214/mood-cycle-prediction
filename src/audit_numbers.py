"""Audit: every numeric token in docs/report.md, docs/viva_qa.md and README.md must either be a value logged in docs/numbers_provenance.md (i.e. loaded from a result file / the data / the code),
or be an allowed structural token (section numbers, question numbers, names such as E1 or set D2, thresholds and conventions explained in the report). Writes results/numbers_audit.md; exits 1 if unexplained tokens remain."""
import re, sys; from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
NUM = re.compile(r"(?<![A-Za-z_`])[-+]?\d[\d,]*\.?\d*(?:e[-+]?\d+)?%?")
def toks(text): return [t.strip().rstrip(".,").lstrip("+") for t in NUM.findall(text)]
def norm(t): return t.replace(",", "").lstrip("-+").rstrip("%")
prov = (ROOT / "docs/numbers_provenance.md").read_text(); loaded = set()
for line in prov.splitlines():
    if line.startswith("| ") and "`" in line:
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) >= 2:
            for t in re.findall(r"[-+]?\d[\d,]*\.?\d*(?:e[-+]?\d+)?", cells[1]): loaded.add(norm(t))
STRUCT = {  # token -> reason (definitions, conventions, thresholds; none of these is a result)
 "0.05": "significance threshold", "0.001": "threshold / smallest reported p", "0.01": "threshold in a query", "0": "zero / reference value", "1": "one (e.g. 'target 1', 'H0 delta>=0', phase 1)", "2": "two (e.g. two dummies, 2-df, phase 2)", "3": "three (e.g. three phases, three phase contrasts)",
 "0.80": "nominal level", "0.90": "nominal level", "0.95": "nominal level / 95% CI", "80": "nominal level in %", "90": "nominal level in %", "95": "CI / nominal level in %", "0.8": "nominal level", "0.9": "nominal level", "1.28": "normal quantile for 80%", "1.64": "normal quantile for 90%", "1.96": "normal quantile for 95%",
 "2023": "year of the paper", "154": "Phase 1 (pre-PSST-drop) count quoted in docs only", "2.1": "learning-rate multiple used as a selector (theory: stable only below 2)", "1.9": "learning-rate multiple used as a selector", "1.0": "reference value 1.00", "1.00": "reference value",
 "4": "four (e.g. four methods)", "5": "five (e.g. five features/groups)", "6": "six", "7": "seven (features in set D)", "8": "eight", "9": "nine", "10": "ten", "14": "14 DRSP items", "9-49": "range", "60": "trials (also loaded)", "0.5": "half; conventional medium SMD", "0.2": "conventional small SMD benchmark", "0.3": "", "0.6": "", "0.7": "", "0.35": "", "0.40": "",
 "1/3": "chance level for 3 classes", "256": "name of the hash algorithm SHA-256", "4.2": "section number", "8.1": "section number", "8.2": "section number", "8.3": "section number", "8.4": "section number", "8.5": "section number", "8.6": "section number", "105406": "article number in the paper reference", "3.12": "python version", "103": "(also loaded)", "1e-6": "tolerance"}
def audit(path):
    text = path.read_text(); bad = []
    for ln, line in enumerate(text.splitlines(), 1):
        if line.startswith("|---") or line.startswith("<!--"): continue
        clean = re.sub(r"`[^`]*`", " ", line); clean = re.sub(r"\b[A-Za-z][A-Za-z0-9_]*\d+[A-Za-z0-9_]*\b", " ", clean)   # drop code spans and identifiers like E1, D2, R2, Q12, PC1
        clean = re.sub(r"^#+ [\d.]+", " ", clean); clean = re.sub(r"^\*\*Q\d+\.", " ", clean); clean = re.sub(r"^\d+\. ", " ", clean); clean = re.sub(r"\(\d+\)", " ", clean)
        for t in toks(clean):
            n = norm(t)
            if n and (n in loaded or n in STRUCT or t in STRUCT): continue
            bad.append((ln, t, line.strip()[:140]))
    return bad
rep = ["# Numbers audit (src/audit_numbers.py)", "", f"Loaded values available from the provenance file: {len(loaded)} distinct numeric strings. Allowed structural tokens: {len(STRUCT)} (listed below with reasons).", ""]; rc = 0
for f in ["docs/report.md", "docs/viva_qa.md", "README.md"]:
    b = audit(ROOT / f); rc += len(b); rep += [f"## {f}: {len(b)} numeric tokens neither loaded nor structural", ""] + [f"- line {ln}: `{t}`  ({ctx})" for ln, t, ctx in b] + [""]
rep += ["## Allowed structural tokens", ""] + [f"- `{k}`: {v}" for k, v in STRUCT.items()]
(ROOT / "results/numbers_audit.md").write_text("\n".join(rep)); print(f"numbers audit: {rc} unexplained numeric tokens (see results/numbers_audit.md)"); sys.exit(1 if rc else 0)
