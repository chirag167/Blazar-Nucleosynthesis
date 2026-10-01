#!/usr/bin/env python3
"""
collapse_pd.py
==============

Collapse the several p+d run directories that all describe the *same* physical
reaction -- the deuteron breakup p + d -> p + p + n (paper notation p(d,pn)p,
a.k.a. "p+d inelastic") -- into a single canonical folder.

Default action is DIAGNOSE ONLY: it reports row counts and groups folders whose
(E, sigma) data are identical, then prints the command that would collapse them.
Nothing on disk changes unless you pass --apply.

When --apply is given it:
  * builds the canonical folder <runs-dir>/<name>/<name>_total_xs.csv,
  * fills it with the rows of --keep plus any *unique* rows from --merge
    (dedup key = (E_MeV, sigma_mb, source); FRESCO rows preferred on ties),
  * moves every listed source folder into <runs-dir>/_archive_pd/ (move, not
    delete -- reversible).

Examples
--------
    # just look
    python collapse_pd.py --runs-dir newruns

    # collapse: keep p_d_inel's data, fold in unique rows from the garbled trio,
    # write canonical folder p_d_pn_p, archive the originals
    python collapse_pd.py --runs-dir newruns --apply \
        --keep p_d_inel --merge p_d_dp_n p_d_n_d p_d_n_p --name p_d_pn_p

Remember to (1) add  "<name>": r"p(d,pn)p"  to REACTION_LABELS and remove the
garbled keys, and (2) point integrate_xs.py / the network at the single folder
so the breakup rate is not multiply counted.
"""

import argparse
import csv
import hashlib
import shutil
from pathlib import Path

DEFAULT_FOLDERS = ["p_d_dp_n", "p_d_n_d", "p_d_n_p", "p_d_inel"]
FIELDS = ["E_MeV", "sigma_mb", "d_sigma_mb", "coverage_frac", "source"]


def csv_path(runs_dir, folder):
    return runs_dir / folder / f"{folder}_total_xs.csv"


def read_rows(path):
    if not path.exists():
        return None
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def data_hash(rows):
    """Hash of the (E, sigma) content, order-independent."""
    pairs = sorted((r.get("E_MeV", ""), r.get("sigma_mb", "")) for r in rows)
    h = hashlib.md5()
    for e, s in pairs:
        h.update(f"{e}|{s};".encode())
    return h.hexdigest()[:10]


def diagnose(runs_dir, folders):
    present, groups = [], {}
    print(f"\nScanning {runs_dir} for p+d folders:\n")
    for folder in folders:
        rows = read_rows(csv_path(runs_dir, folder))
        if rows is None:
            print(f"  {folder:12s}  (no CSV found -- skipping)")
            continue
        h = data_hash(rows)
        n_fresco = sum(1 for r in rows if r.get("source") == "fresco")
        n_exp = len(rows) - n_fresco
        present.append(folder)
        groups.setdefault(h, []).append(folder)
        print(f"  {folder:12s}  rows={len(rows):3d}  "
              f"(fresco={n_fresco}, exp={n_exp})  data-hash={h}")

    dup_groups = [g for g in groups.values() if len(g) > 1]
    print("\nDuplicate groups (identical (E, sigma) data):")
    if dup_groups:
        for g in dup_groups:
            print(f"  {' == '.join(g)}")
    else:
        print("  none -- folders hold distinct data; union them rather than delete.")

    return present


def collapse(runs_dir, keep, merge, name, apply):
    keep_rows = read_rows(csv_path(runs_dir, keep))
    if keep_rows is None:
        raise SystemExit(f"--keep folder {keep!r} has no CSV at "
                         f"{csv_path(runs_dir, keep)}")

    seen = {(r.get("E_MeV"), r.get("sigma_mb"), r.get("source")) for r in keep_rows}
    merged = list(keep_rows)
    added = 0
    for folder in merge:
        rows = read_rows(csv_path(runs_dir, folder))
        if rows is None:
            print(f"  [warn] merge source {folder!r} has no CSV -- skipped")
            continue
        for r in rows:
            key = (r.get("E_MeV"), r.get("sigma_mb"), r.get("source"))
            if key not in seen:
                seen.add(key)
                merged.append(r)
                added += 1

    merged.sort(key=lambda r: (float(r.get("E_MeV") or "inf"), r.get("source", "")))
    print(f"\nCanonical folder: {name}")
    print(f"  base rows from {keep}: {len(keep_rows)}")
    print(f"  unique rows added from {merge}: {added}")
    print(f"  total rows: {len(merged)}")

    sources_to_archive = [keep] + list(merge)
    if not apply:
        print("\n(dry run -- nothing written. Re-run with --apply to execute.)")
        print("Folders that would be archived to _archive_pd/:",
              ", ".join(sources_to_archive))
        return

    out_dir = runs_dir / name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{name}_total_xs.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in merged:
            w.writerow({k: r.get(k, "") for k in FIELDS})
    print(f"  wrote {out_csv}")

    archive = runs_dir / "_archive_pd"
    archive.mkdir(exist_ok=True)
    for folder in sources_to_archive:
        src = runs_dir / folder
        if src.resolve() == out_dir.resolve():
            continue  # don't archive the folder we just wrote into
        if src.exists():
            shutil.move(str(src), str(archive / folder))
            print(f"  archived {folder} -> _archive_pd/{folder}")

    print(f"\nDone. Now edit REACTION_LABELS: add \"{name}\": r\"p(d,pn)p\" "
          f"and remove the garbled p_d_* keys.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", default="newruns")
    ap.add_argument("--folders", nargs="*", default=DEFAULT_FOLDERS,
                    help="candidate p+d folders to inspect")
    ap.add_argument("--keep", help="folder whose data becomes the base")
    ap.add_argument("--merge", nargs="*", default=[],
                    help="extra folders whose unique rows to fold in")
    ap.add_argument("--name", default="p_d_pn_p",
                    help="canonical folder name to create (default: p_d_pn_p)")
    ap.add_argument("--apply", action="store_true",
                    help="actually write the canonical folder and archive sources")
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    present = diagnose(runs_dir, args.folders)

    if not args.keep:
        print("\nNo --keep given, so this was diagnosis only.")
        print("Pick the folder with the real excitation function (Fresco curve +")
        print("points, not a flat plateau) and re-run, e.g.:")
        example_keep = "p_d_inel" if "p_d_inel" in present else (present[0] if present else "p_d_inel")
        others = [f for f in present if f != example_keep]
        print(f"\n  python collapse_pd.py --runs-dir {args.runs_dir} --apply \\")
        print(f"      --keep {example_keep} --merge {' '.join(others)} --name {args.name}")
        return

    merge = [f for f in args.merge if f != args.keep]
    collapse(runs_dir, args.keep, merge, args.name, args.apply)


if __name__ == "__main__":
    main()
