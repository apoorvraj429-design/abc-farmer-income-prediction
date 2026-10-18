"""CSV parsing and row-specific validation around the unchanged model contract."""
import csv
import io
import numpy as np
import pandas as pd
from core import FEATURES, NUM, LABELS, extract_features, validate_inputs

MAX_BYTES = 25 * 1024 * 1024
MAX_ROWS = 15000


class BatchInputError(ValueError):
    pass


def read_batch(upload):
    """Reject oversized/malformed inputs before scoring. Keep original row positions."""
    data = upload.getvalue() if hasattr(upload, "getvalue") else upload.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise BatchInputError("Use a CSV of 25 MB or less.")
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise BatchInputError("Save the file as CSV UTF-8 and upload it again.") from exc
    try:
        headers = next(csv.reader(io.StringIO(text)))
    except (StopIteration, csv.Error) as exc:
        raise BatchInputError("The CSV is empty or its header cannot be read.") from exc
    stripped = [h.strip() for h in headers]
    if len(stripped) != len(set(stripped)):
        raise BatchInputError("Duplicate column names are not allowed. Check the CSV header.")
    try:
        frame = pd.read_csv(io.StringIO(text), low_memory=False, nrows=MAX_ROWS + 1)
    except pd.errors.EmptyDataError as exc:
        raise BatchInputError("The CSV contains no data.") from exc
    if len(frame) > MAX_ROWS:
        raise BatchInputError("Use at most 15,000 records per upload.")
    return frame


def inspect_rows(frame, meta):
    """Report all invalid positions, then defer valid processing to core.validate_inputs."""
    if frame.empty:
        raise BatchInputError("The file contains no records.")
    if len(frame) > MAX_ROWS:
        raise BatchInputError("Use at most 15,000 records per upload.")
    x = extract_features(frame).reset_index(drop=True)
    issues = []

    def add(mask, field, reason):
        for pos in np.flatnonzero(np.asarray(mask)):
            issues.append({"source_row": int(pos + 1), "field": field,
                           "value": "" if pd.isna(x.at[pos, field]) else str(x.at[pos, field]),
                           "issue": reason})

    add(x.state.eq(""), "state", "State is required.")
    add(x.state.ne("") & ~x.state.isin(meta["states"]), "state", "Unsupported state. Use a state from the app list.")
    for field in NUM:
        original = x[field]
        values = pd.to_numeric(original, errors="coerce")
        malformed = values.isna() & original.notna() & original.astype(str).str.strip().ne("")
        add(malformed, field, LABELS[field] + " must be numeric.")
        add(values.notna() & (~np.isfinite(values) | values.lt(0)), field,
            "Use a finite, non-negative number.")
        if field == "non_ag_income":
            add(values.isna() & ~malformed, field,
                "Known non-farm income is required. Zero means truly zero, not unknown.")
    issue_frame = pd.DataFrame(issues, columns=["source_row", "field", "value", "issue"])
    if issues:
        return x, issue_frame.sort_values(["source_row", "field"]).reset_index(drop=True), []
    valid, warnings = validate_inputs(frame, meta)
    return valid, issue_frame, warnings
