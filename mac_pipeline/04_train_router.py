"""
mac_pipeline/04_train_router.py
-------------------------------
Train split only.

  1. Frozen MiniLM embeddings for EVERY question (train and test), tracked by CodeCarbon so
     the router's own energy per question is known and charged later.
  2. "Whether": sweep tau on the base model's confidence (out-of-fold on train).
  3. "Where": per pool model, logistic regression on [base confidence, PCA-32(embedding)]
     -> P(pool model right). Regularisation by 5-fold CV. Also a subject-one-hot baseline.
  4. Save router/embeddings.npy, router/router.pkl, router/thresholds.json,
     results/router_train.md (OOF AUCs and the tau chosen).

  python mac_pipeline/04_train_router.py [--no-energy]
"""

import argparse
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from common import energy
from common.data import load_questions, load_scores, pivot_scores

C_GRID = (0.1, 0.3, 1.0, 3.0)


def embed_all(q: pd.DataFrame, track: bool):
    from sentence_transformers import SentenceTransformer
    texts = q["prompt_text"].tolist()
    if track:
        idle = energy.idle_baseline(min(config.IDLE_SECS, 30), "router_embed")
        with energy.tracked("router_embed") as run:
            E = SentenceTransformer(config.ROUTER_EMBEDDER).encode(
                texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
        summ = energy.summarize_run(run, role="router", model_id=config.ROUTER_EMBEDDER,
                                    dataset="all", repeat=0, n=len(texts), idle=idle)
        rec = energy.record_kwh(summ)
        summ["kwh_per_question"] = rec / len(texts)
        energy.write_summary(summ, "router_embed")
        print(f"embedding energy: {rec * 1000:.4f} Wh for {len(texts)} q  "
              f"({summ['kwh_per_question'] * 3.6e6:.3f} J/q)")
    else:
        E = SentenceTransformer(config.ROUTER_EMBEDDER).encode(
            texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
    np.save(config.ROUTER_DIR / "embeddings.npy", E)
    pd.Series(q["id"]).to_csv(config.ROUTER_DIR / "embedding_ids.csv", index=False)
    return E


def make_head(name: str, seed: int = 0, C: float = 0.5):
    """Router head = a probabilistic classifier on [confidence, PCA(embedding)].

    No class_weight anywhere: the router picks argmax P(right) across pool models, so each
    model's base rate must survive -- 'balanced' erased exactly the dominance information.
    lr is the literature default for dense embeddings; the others are the classifier families
    the paper used on the 8 NeMo features, kept as an ablation (04 compares them out-of-fold).
    """
    if name == "lr":
        return LogisticRegression(max_iter=4000, C=C)
    if name == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=seed, n_jobs=-1)
    if name == "xgb":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                             colsample_bytree=0.8, reg_lambda=1.0, random_state=seed, n_jobs=-1,
                             eval_metric="logloss", verbosity=0)
    if name == "svc":
        from sklearn.svm import SVC
        return SVC(C=1.0, kernel="rbf", probability=True, random_state=seed)
    if name == "mlp":
        from sklearn.neural_network import MLPClassifier
        return MLPClassifier(hidden_layer_sizes=(64,), alpha=1e-2, max_iter=1000, early_stopping=True,
                             random_state=seed)
    if name == "voting":
        from sklearn.ensemble import VotingClassifier
        return VotingClassifier([("lr", make_head("lr", seed, C)), ("rf", make_head("rf", seed)),
                                 ("xgb", make_head("xgb", seed))], voting="soft")
    raise ValueError(name)


def oof_auc(X, y, seed, C=0.5, head="lr"):
    if y.sum() < 5 or (1 - y).sum() < 5:
        return np.nan, np.full(len(y), y.mean())
    p = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        sc = StandardScaler().fit(X[tr])
        m = make_head(head, seed, C).fit(sc.transform(X[tr]), y[tr])
        p[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
    return roc_auc_score(y, p), p


def oof_pca_features(E, conf, seed):
    """PCA fitted inside each fold so the embedding never sees held-out rows."""
    # For simplicity and speed the PCA is fitted on all train rows here; the CV is over the
    # logistic regression. PCA is unsupervised so this is a mild leak at most.
    Z = PCA(config.PCA_DIMS, random_state=seed).fit_transform(E)
    return np.hstack([conf.reshape(-1, 1), Z])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-energy", action="store_true")
    args = ap.parse_args()

    q = load_questions()
    E_all = embed_all(q, track=not args.no_energy)
    id2row = {i: k for k, i in enumerate(q["id"])}

    scores = load_scores()
    roles = [r for r in config.ALL_ROLES if r in set(scores["model"])]
    pool = [r for r in config.POOL if r in roles]
    piv = pivot_scores(scores, roles, fold="train")
    ids = piv.index.tolist()
    n = len(ids)
    E = E_all[[id2row[i] for i in ids]]
    conf = piv[("graded", config.BASE)].values
    C = {r: piv[("correct", r)].values.astype(int) for r in roles}
    subj = q.set_index("id").loc[ids, "subject"]
    S = pd.get_dummies(subj).values.astype(float)
    print(f"train n={n}  roles={roles}")

    lines = ["# Router training (train split, out-of-fold)", "", f"n = {n}", ""]

    # ---- "whether": is the base wrong?  confidence vs embedding vs both
    y_wrong = 1 - C[config.BASE]
    Xc = conf.reshape(-1, 1)
    Xpca = oof_pca_features(E, conf, 0)
    aucs = {}
    for name, X in (("confidence", Xc), ("embedding PCA32", Xpca[:, 1:]), ("conf + PCA32", Xpca),
                    ("subject one-hot", S)):
        aucs[name] = np.nanmean([oof_auc(X, y_wrong, s)[0] for s in config.ROUTER_SEEDS])
    lines += ["## Target: base is wrong", "", "| features | OOF AUC |", "|---|---:|"]
    lines += [f"| {k} | {v:.3f} |" for k, v in aucs.items()]
    print("base-wrong AUC:", {k: round(v, 3) for k, v in aucs.items()})

    # ---- "where": per pool model P(right). Feature ablation with lr, then head ablation.
    lines += ["", "## Target: pool model is right (lr head, feature ablation)", "",
              "| pool model | conf | PCA32 | conf+PCA32 | subject | chosen C |", "|---|---:|---:|---:|---:|---:|"]
    best_C = {}
    for r in pool:
        y = C[r]
        a_c = np.nanmean([oof_auc(Xc, y, s)[0] for s in config.ROUTER_SEEDS])
        a_e = np.nanmean([oof_auc(Xpca[:, 1:], y, s)[0] for s in config.ROUTER_SEEDS])
        a_s = np.nanmean([oof_auc(S, y, s)[0] for s in config.ROUTER_SEEDS])
        best_c, best_a = None, -1
        for c in C_GRID:
            a = np.nanmean([oof_auc(Xpca, y, s, C=c)[0] for s in config.ROUTER_SEEDS])
            if a > best_a:
                best_c, best_a = c, a
        best_C[r] = best_c
        lines.append(f"| {r} | {a_c:.3f} | {a_e:.3f} | {best_a:.3f} | {a_s:.3f} | {best_c} |")
        print(f"{r}: conf {a_c:.3f}  emb {a_e:.3f}  both {best_a:.3f}  subject {a_s:.3f}  C={best_c}")

    # head ablation: same features, every classifier family; OOF AUC per pool model and the OOF
    # hybrid accuracy the head's argmax picks produce at tau = 1 (every question routed)
    stack = np.vstack([C[r] for r in pool])
    heads = [h for h in config.ROUTER_HEADS]
    lines += ["", "## Head ablation (conf + PCA32 features, OOF)", "",
              "| head | " + " | ".join(f"AUC {r}" for r in pool) + " | mean AUC | routed acc (all escalated) | oracle |",
              "|---|" + "---:|" * (len(pool) + 3)]
    head_res = {}
    for h in heads:
        try:
            P_h, A_h = [], []
            for r in pool:
                res = [oof_auc(Xpca, C[r], s, C=best_C[r], head=h) for s in config.ROUTER_SEEDS]
                A_h.append(np.nanmean([x[0] for x in res]))
                P_h.append(np.mean([x[1] for x in res], axis=0))
            P_h = np.vstack(P_h)
            routed = stack[P_h.argmax(axis=0), np.arange(n)].mean()
            head_res[h] = dict(aucs=A_h, mean_auc=float(np.nanmean(A_h)), P=P_h, routed=float(routed))
            lines.append(f"| {h} | " + " | ".join(f"{a:.3f}" for a in A_h) +
                         f" | {head_res[h]['mean_auc']:.3f} | {routed:.3f} | {stack.max(axis=0).mean():.3f} |")
            print(f"head {h:<7} AUC {[round(a, 3) for a in A_h]}  mean {head_res[h]['mean_auc']:.3f}  routed {routed:.3f}")
        except Exception as e:                      # e.g. xgboost not installed
            lines.append(f"| {h} | " + " | ".join("-" for _ in pool) + f" | - | - | ({type(e).__name__}) |")
            print(f"head {h}: skipped ({type(e).__name__}: {e})")
    if config.ROUTER_HEAD == "auto":
        # select on the objective the router is used for -- OOF accuracy of its argmax picks --
        # not on AUC: on the first Mac run rf had the best AUC and the worst routed accuracy
        head = max(head_res, key=lambda h: (round(head_res[h]["routed"], 4), head_res[h]["mean_auc"]))
    else:
        head = config.ROUTER_HEAD
        if head not in head_res:
            sys.exit(f"router head {head} did not train")
    lines += ["", f"**Head used: `{head}`** ({'best OOF routed accuracy, ties by mean AUC' if config.ROUTER_HEAD == 'auto' else 'fixed in config'})."]
    print(f"router head: {head}")

    # fit the chosen head on all train rows, per pool model
    router = {"pool": pool, "pca_dims": config.PCA_DIMS, "head": head, "models": {}}
    for j, r in enumerate(pool):
        pca = PCA(config.PCA_DIMS, random_state=0).fit(E)
        X = np.hstack([conf.reshape(-1, 1), pca.transform(E)])
        sc = StandardScaler().fit(X)
        clf = make_head(head, 0, best_C[r]).fit(sc.transform(X), C[r])
        router["models"][r] = {"pca": pca, "scaler": sc, "clf": clf, "C": best_C[r],
                               "oof_auc": head_res[head]["aucs"][j]}
    with open(config.ROUTER_DIR / "router.pkl", "wb") as f:
        pickle.dump(router, f)

    # ---- tau sweep on train (OOF router picks of the chosen head), to choose tau*
    P = head_res[head]["P"]
    choice = P.argmax(axis=0)
    best_single = max(roles, key=lambda r: C[r].mean())
    lines += ["", f"## tau sweep on train (hybrid = base -> router pick; plain = base -> {best_single})", "",
              "| tau | escalated | hybrid acc | plain acc | oracle-router acc |", "|---:|---:|---:|---:|---:|"]
    sweep = []
    for tau in config.TAU_GRID:
        esc = conf < tau
        hyb = np.where(esc, stack[choice, np.arange(n)], C[config.BASE]).mean()
        plain = np.where(esc, C[best_single], C[config.BASE]).mean()
        orc = np.where(esc, stack.max(axis=0), C[config.BASE]).mean()
        sweep.append(dict(tau=tau, escalated=float(esc.mean()), hybrid=float(hyb), plain=float(plain), oracle=float(orc)))
        lines.append(f"| {tau:.2f} | {esc.mean():.1%} | {hyb:.3f} | {plain:.3f} | {orc:.3f} |")
    # tau*: best hybrid accuracy; ties -> least escalation
    inner = [s for s in sweep if 0 < s["escalated"] < 1]
    tau_star = max(inner or sweep, key=lambda s: (round(s["hybrid"], 4), -s["escalated"]))["tau"]
    thresholds = {"tau_star": tau_star, "best_single": best_single, "sweep": sweep,
                  "base_wrong_auc": aucs, "pool": pool, "head": head,
                  "head_ablation": {h: {"aucs": v["aucs"], "mean_auc": v["mean_auc"], "routed": v["routed"]}
                                    for h, v in head_res.items()}}
    (config.ROUTER_DIR / "thresholds.json").write_text(json.dumps(thresholds, indent=2))
    lines += ["", f"**tau\\* = {tau_star}** (best OOF hybrid accuracy on train, ties to least escalation)"]
    (config.RESULTS_DIR / "router_train.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\ntau* = {tau_star}   wrote router/router.pkl, router/thresholds.json, results/router_train.md")


if __name__ == "__main__":
    main()
