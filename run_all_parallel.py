"""Driver (not part of the paper code): runs step6.run_dataset for all 20
datasets using 2 worker processes; per-dataset console output -> logs/<name>.log,
result dicts -> step6_run_results.csv"""
import os
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["OPENBLAS_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"
import sys, csv, contextlib, time
from multiprocessing import Pool

ORDER = ["Adiac", "StarLightCurves", "Lighting7", "OSULeaf", "Computers", "Wafer", "Earthquakes",
         "CinC_ECG_torso", "CBF", "Trace", "ArrowHead", "ECG200", "Gun_Point", "ItalyPowerDemand",
         "Beef", "OliveOil", "DiatomSizeReduction", "BirdChicken", "ECGFiveDays", "Coffee"]

def work(name):
    import step6
    t0 = time.time()
    with open(f"logs/{name}.log", "w", buffering=1) as f, contextlib.redirect_stdout(f):
        r = step6.run_dataset(name)
    r["wall_s"] = round(time.time() - t0, 1)
    return r

if __name__ == "__main__":
    import step6
    assert sorted(ORDER) == sorted(step6.DATASETS)
    rows = []
    with Pool(2) as p:
        for r in p.imap_unordered(work, ORDER):
            print(time.strftime("%H:%M:%S"), r.get("dataset"), r.get("status"),
                  "attn=%s ed=%s dtw=%s wall=%ss" % (r.get("attn_acc"), r.get("ed_acc"), r.get("dtw_acc"), r.get("wall_s")), flush=True)
            rows.append(r)
            keys = sorted({k for x in rows for k in x})
            with open("step6_run_results.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print("ALL DONE", flush=True)
