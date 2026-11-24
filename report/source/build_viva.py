"""Fill viva_template.md with the trained models' numbers -> report/Viva_Prep.md"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
mri = json.load(open(os.path.join(ROOT, "results", "mri", "metrics.json")))
slp = json.load(open(os.path.join(ROOT, "results", "sleep", "metrics.json")))["results"]
facts = json.load(open(os.path.join(HERE, "facts.json")))
R, B, dd = mri["results"][mri["chosen"]], mri["results"]["neurova_cnn"], mri["dedup"]
mis = mri.get("test_mistakes", [])
p = lambda x: f"{x * 100:.1f}%"
vals = {
    "removed": dd["images_total"] - dd["unique_images"], "total": f"{dd['images_total']:,}",
    "cross": dd["groups_spanning_official_train_and_test"], "cnn_acc": p(B["test_accuracy"]), "eff_acc": p(R["test_accuracy"]),
    "temp": f"{R['temperature']:.2f}", "ece_b": f"{R['ece_before']:.3f}", "ece_a": f"{R['ece_after']:.3f}",
    "mistakes": len(mis), "n_test": sum(mri["splits"]["test"].values()),
    "missed": sum(1 for m in mis if m["true"] != "notumor" and m["pred"] == "notumor"),
    "kappa": f"{slp['test_kappa']:.2f}", "n1_f1": f"{slp['per_class']['N1']['f1']:.2f}", "w_rec": p(slp["per_class"]["W"]["recall"]),
    "latency": round(facts["latency"]["forward_ms"]), "req": round(facts["latency"]["mri_request_s"], 1),
    **(lambda M: (lambda G, T: {
        "gru_acc": f"{G['test_accuracy']['mean'] * 100:.1f}%", "trf_acc": f"{T['test_accuracy']['mean'] * 100:.1f}%",
        "gru_rng": f"{G['test_accuracy']['min'] * 100:.1f} to {G['test_accuracy']['max'] * 100:.1f}%",
        "trf_rng": f"{T['test_accuracy']['min'] * 100:.1f} to {T['test_accuracy']['max'] * 100:.1f}%",
        "gru_val": f"{G['best_val_macro_f1']['mean']:.3f}", "trf_val": f"{T['best_val_macro_f1']['mean']:.3f}",
        "trf_first": f"{T['runs'][0]['test_accuracy'] * 100:.1f}%", "gru_first": f"{G['runs'][0]['test_accuracy'] * 100:.1f}%",
        "trf_verdict": ("So the first Transformer result was luck: the GRU is better on average and far steadier, and the Transformer's validation score swung between epochs."
                        if G["test_accuracy"]["mean"] > T["test_accuracy"]["mean"] else
                        "So the Transformer held up over the three runs."),
    })(*[next(x for x in M if x["key"] == k) for k in ("gru", "transformer")]))(
        json.load(open(os.path.join(ROOT, "results", "sleep", "comparison.json")))["models"]),
    **(lambda F: {"fam_k": F["pca_dimensions"], "fam_caught": F["test_mistakes_flagged"], "fam_false": F["ordinary_test_scans_flagged_percent"],
                  "fam_updown": F["upside_down_scans_flagged_percent"]})(json.load(open(os.path.join(ROOT, "results", "mri", "familiarity.json")))),
    "eff_acc_deployed": f"{R['test_accuracy'] * 100:.1f}%",
    "flip_std": f"{mri['results']['efficientnet_b0_tf']['robustness']['noise_0.05']:.0f}%",
    "flip_rob": f"{mri['results'].get('efficientnet_b0_tf_robust', R)['robustness']['noise_0.05']:.1f}%",
    "rob_acc": p(mri['results'].get('efficientnet_b0_tf_robust', R)['test_accuracy']),
    "deploy_note": ("It met the rule, so it is the deployed model." if mri["chosen"] == "efficientnet_b0_tf_robust"
                    else "It didn't meet the rule, so the standard model is deployed and the fragility is listed as a limitation."),
    "missed_text": (lambda miss: "No tumour was called normal." if not miss else
                    f"One tumour was called no tumour, at {miss[0]['confidence'] * 100:.0f}% confidence, which the uncertainty display would flag."
                    if len(miss) == 1 else f"{len(miss)} tumours were called no tumour.")(
                    [m for m in mis if m["true"] != "notumor" and m["pred"] == "notumor"]),
    "price_mri": facts["business"]["price_per_mri"], "clinic": f"{facts['business']['clinic_plan_monthly']:,}",
}
text = open(os.path.join(HERE, "viva_template.md"), encoding="utf-8").read()
for k, v in vals.items():
    text = text.replace("{" + k + "}", str(v))
os.makedirs(os.path.join(ROOT, "report"), exist_ok=True)
open(os.path.join(ROOT, "report", "Viva_Prep.md"), "w", encoding="utf-8").write(text)
print("wrote Viva_Prep.md")
