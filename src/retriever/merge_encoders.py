"""Build the final retriever M2' by averaging the weights of the DocID encoder and the retriever.

    theta' = alpha * theta_1 + (1 - alpha) * theta_2      every float tensor of the encoder
    w'     = alpha * 1       + (1 - alpha) * w            the per-bit weights

The result is saved in the retriever's checkpoint format. DocIDs are not changed.

    python merge_encoders.py --stage1 <docid_encoder.pt> --stage2 <retriever.pt> --out <merged.pt>
"""
import argparse, json, os, time
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", required=True)
    ap.add_argument("--stage2", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", type=float, default=0.5, help="weight on the DocID encoder.")
    a = ap.parse_args()
    if os.path.exists(a.out):
        raise SystemExit(f"REFUSING to overwrite {a.out}")

    sd1 = {k: v.detach().cpu() for k, v in torch.load(a.stage1, map_location="cpu").items()}
    blob = torch.load(a.stage2, map_location="cpu")
    if set(blob["head"]) != {"zs"}:
        raise SystemExit("the retriever checkpoint must hold per-bit weights only")
    sd2 = {k: v.detach().cpu() for k, v in blob["enc"].items()}
    if set(sd1) != set(sd2):
        raise SystemExit("stage-1 and stage-2 encoders have different parameters")
    al = a.alpha
    merged = {k: ((al * sd1[k].float() + (1 - al) * sd2[k].float()).to(sd1[k].dtype)
                  if sd1[k].is_floating_point() else sd2[k]) for k in sd1}
    head = {k: v.clone() for k, v in blob["head"].items()}
    head["zs"] = al * torch.ones_like(blob["head"]["zs"].float()) + (1 - al) * blob["head"]["zs"].float()

    out = dict(blob)
    out["enc"], out["head"] = merged, head
    out["fusion"] = {"alpha": al, "stage1": os.path.abspath(a.stage1),
                     "stage2": os.path.abspath(a.stage2),
                     "made": time.strftime("%Y-%m-%d %H:%M:%S"),
                     "note": "weight fusion (WiSE-FT style); identifiers unchanged (stage-1 codes)"}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    torch.save(out, a.out)
    # Copy the architecture sidecar from the DocID encoder.
    side1, side = a.stage1 + ".arch.json", a.out + ".arch.json"
    if os.path.exists(side1) and not os.path.exists(side):
        arch = json.load(open(side1))
        arch["fusion_alpha"] = al
        json.dump(arch, open(side, "w"), indent=1)
    d = max((merged[k].float() - sd1[k].float()).abs().max().item()
            for k in merged if merged[k].is_floating_point())
    print(f"wrote {a.out}  alpha={al}  |theta-theta1|max={d:.4f}  zs mean {head['zs'].mean():.4f}")


if __name__ == "__main__":
    main()
