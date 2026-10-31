import csv
from pathlib import Path


def test_step5b_references_come_from_step4_files():
    folder = Path(__file__).resolve().parent.parent / "week6/noise_reachability_step5b_colab/references"
    base = list(csv.DictReader(open(folder / "step4_base_per_episode.csv")))
    collection = list(csv.DictReader(open(folder / "step4_collection.csv")))
    assert (len(base), sum(int(r["success"]) for r in base)) == (160, 37)
    assert (len(collection), sum(int(r["success"]) for r in collection)) == (120, 50)
