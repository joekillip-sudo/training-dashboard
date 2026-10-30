"""python -m tennis.cli fetch|report|demo"""
import argparse
import json
from pathlib import Path

from . import data, report, synth


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="download the yearly ATP/WTA files")
    f.add_argument("--out", default="tennis/data_raw")
    r = sub.add_parser("report", help="run the walk-forward backtest and write the report")
    r.add_argument("--data", default="tennis/data_raw")
    r.add_argument("--out", default="tennis_report.txt")
    r.add_argument("--test-start", default=report.TEST_START)
    sub.add_parser("demo", help="run the report on generated data (no network)")
    a = ap.parse_args(argv)
    if a.cmd == "fetch":
        man = data.fetch_all(a.out)
        print(json.dumps(man, indent=1))
        print(f"{len(man)} files in {a.out}")
        if not man:
            raise SystemExit("nothing downloaded")
    elif a.cmd == "report":
        matches, diags = data.load_dir(a.data)
        notes = Path(a.data) / "notes.txt"
        text = report.build_report(matches, diags, a.test_start)
        if notes.exists():
            text += "\nSITE NOTES (tennis-data.co.uk/notes.txt, verbatim):\n" + notes.read_bytes().decode("latin-1") + "\n"
        Path(a.out).write_text(text)
        print(text)
    else:
        m, d = [], []
        for tour, sd in (("ATP", 0.0), ("WTA", 0.0)):
            mm, dd = data.clean(synth.synth_raw(noise_sd=sd, seed=5 if tour == "ATP" else 6), tour)
            dd.update(file="synthetic", tour=tour, year=0, kept=len(mm))
            m.append(mm)
            d.append(dd)
        import pandas as pd
        print(report.build_report(pd.concat(m, ignore_index=True), d, "2010-01-01"))


if __name__ == "__main__":
    main()
