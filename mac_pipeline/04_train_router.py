"""
mac_pipeline/04_train_router.py
-------------------------------
Train split only. The decision rule is route-then-cascade:

  router (ex-ante, from the question text alone)  ->  one of the ROUTED 1.5B models answers
  that model's confidence < tau[route]            ->  the ESCALATION 3B model answers instead

  1. Frozen MiniLM embeddings for EVERY question (train and test), tracked so the router's own
     energy per question is known and charged later.
  2. "Where": per routed model, a classifier on PCA-32(embedding) -> P(that model is right).
     The router picks the argmax. Feature ablation (embedding vs subject one-hot), head
     ablation (lr / rf / xgb / svc / mlp / voting), everything out-of-fold on train.
  3. "Whether": per route, sweep tau on the routed model's own confidence (out-of-fold picks)
     and choose tau[route]; also one global tau for the ablation and the tau of the plain
     cascade generalist -> escalation.
  4. Save router/embeddings.npy, router/router.pkl, router/thresholds.json,
     results/router_train.md.

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
    encoder = SentenceTransformer(config.ROUTER_EMBEDDER)
    texts = q["prompt_text"].tolist()
    if track:
        idle = energy.idle_baseline(min(config.IDLE_SECS, 30), "router_embed")
        with energy.tracked("router_embed", precondition_heavy=True) as run:
            E = encoder.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
        summ = energy.summarize_run(run, role="router", model_id=config.ROUTER_EMBEDDER,
                                    dataset="all", repeat=0, n=len(texts), idle=idle)
        rec = energy.record_kwh(summ)
        summ["kwh_per_question"] = rec / len(texts)
        energy.write_summary(summ, "router_embed")
        print(f"embedding energy: {rec * 1000:.4f} Wh for {len(texts)} q  "
              f"({summ['kwh_per_question'] * 3.6e6:.3f} J/q)")
    else:
        E = encoder.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
    np.save(config.ROUTER_DIR / "embeddings.npy", E)
    pd.Series(q["id"]).to_csv(config.ROUTER_DIR / "embedding_ids.csv", index=False)
    return E


def make_head(name: str, seed: int = 0, C: float = 0.5):
    """Router head = a probabilistic classifier on PCA(embedding).

    No class_weight anywhere: the router picks argmax P(right) across the routed models, so
    each model's base rate must survive -- 'balanced' erased exactly the dominance information.
    lr is the literature default for dense embeddings; the others are the classifier families
    the paper used on the 8 NeMo features, kept as an ablation.
    """
    if name == "lr":
        return LogisticRegression(max_iter=4000, C=C)
    if name == "rf":
        from sklearn.ensemble import RandomForestClassifier
        # Keep CV fits single-worker on macOS: joblib process pools can leave leaked
        # semaphores and compete with the native OpenMP runtime used by XGBoost.
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=seed, n_jobs=1)
    if name == "xgb":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                             colsample_bytree=0.8, reg_lambda=1.0, random_state=seed, n_jobs=1,
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
    """5-fold out-of-fold probabilities and AUC. Scaler fitted inside each fold."""
    if y.sum() < 5 or (1 - y).sum() < 5:
        return np.nan, np.full(len(y), y.mean())
    p = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        sc = StandardScaler().fit(X[tr])
        m = make_head(head, seed, C).fit(sc.transform(X[tr]), y[tr])
        p[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
    return roc_auc_score(y, p), p


def pca_features(E, seed=0):
    # PCA is unsupervised and fitted on all train rows (never on test); the CV is over the head
    return PCA(config.PCA_DIMS, random_state=seed).fit_transform(E)


def choose_tau(conf, c_route, c_esc):
    """Sweep tau on one route's confidence: accuracy if conf < tau escalates. Returns the sweep
    and tau*: best accuracy among non-degenerate points, ties to the least escalation."""
    sweep = []
    for tau in config.TAU_GRID:
        esc = conf < tau
        acc = np.where(esc, c_esc, c_route).mean() if len(conf) else np.nan
        sweep.append(dict(tau=tau, escalated=float(esc.mean()) if len(conf) else np.nan, acc=float(acc)))
    if len(conf) == 0:                     # a route the OOF router never picked
        return sweep, None
    inner = [s for s in sweep if 0 < s["escalated"] < 1]
    best = max(inner or sweep, key=lambda s: (round(s["acc"], 4), -s["escalated"]))
    return sweep, best["tau"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-energy", action="store_true")
    args = ap.parse_args()

    q = load_questions()
    E_all = embed_all(q, track=not args.no_energy)
    id2row = {i: k for k, i in enumerate(q["id"])}

    scores = load_scores()
    roles = [r for r in config.ALL_ROLES if r in set(scores["model"])]
    routed = [r for r in config.ROUTED if r in roles]
    esc_role = config.ESCALATION
    if esc_role not in roles or len(routed) < 2:
        sys.exit(f"need scores for {config.ROUTED} and {esc_role}; have {roles}")
    piv = pivot_scores(scores, roles, fold="train")
    ids = piv.index.tolist()
    n = len(ids)
    E = E_all[[id2row[i] for i in ids]]
    C = {r: piv[("correct", r)].values.astype(int) for r in roles}
    CONF = {r: piv[("graded", r)].values for r in roles}
    qi = q.set_index("id").loc[ids]
    subj = qi["subject"]
    ds_of = qi["dataset"].values
    S = pd.get_dummies(subj).values.astype(float)
    Z = pca_features(E)
    stack = np.vstack([C[r] for r in routed])
    print(f"train n={n}  routed={routed}  escalation={esc_role}")

    lines = ["# Router training (train split, out-of-fold)", "",
             f"n = {n}; routed tier = {routed}; escalation = {esc_role}", "",
             "Decision rule: the router reads the question text (MiniLM embedding, PCA-32) and picks "
             "the routed model with the highest predicted P(right). That model answers; if its option "
             "confidence is below the route's tau, the escalation model answers instead.", ""]

    # ---- "where": per routed model P(right) from the text alone. Feature ablation with lr.
    lines += ["## Target: routed model is right (lr head, feature ablation, OOF AUC)", "",
              "| routed model | train acc | PCA32 | subject one-hot | chosen C |", "|---|---:|---:|---:|---:|"]
    best_C = {}
    for r in routed:
        y = C[r]
        a_s = np.nanmean([oof_auc(S, y, s)[0] for s in config.ROUTER_SEEDS])
        best_c, best_a = None, -1
        for c in C_GRID:
            a = np.nanmean([oof_auc(Z, y, s, C=c)[0] for s in config.ROUTER_SEEDS])
            if a > best_a:
                best_c, best_a = c, a
        best_C[r] = best_c
        lines.append(f"| {r} | {y.mean():.3f} | {best_a:.3f} | {a_s:.3f} | {best_c} |")
        print(f"{r}: acc {y.mean():.3f}  AUC emb {best_a:.3f}  subject {a_s:.3f}  C={best_c}")

    # head ablation: OOF routed accuracy of each head's argmax picks, against the singles and oracle
    heads = list(config.ROUTER_HEADS)
    lines += ["", "## Head ablation (PCA32 features, OOF)", "",
              "| head | " + " | ".join(f"AUC {r}" for r in routed) + " | mean AUC | routed acc | " +
              " | ".join(f"always-{r}" for r in routed) + " | oracle |",
              "|---|" + "---:|" * (2 * len(routed) + 3)]
    head_res = {}
    for h in heads:
        try:
            P_h, A_h = [], []
            for r in routed:
                res = [oof_auc(Z, C[r], s, C=best_C[r], head=h) for s in config.ROUTER_SEEDS]
                A_h.append(np.nanmean([x[0] for x in res]))
                P_h.append(np.mean([x[1] for x in res], axis=0))
            P_h = np.vstack(P_h)
            pick = P_h.argmax(axis=0)
            acc = stack[pick, np.arange(n)].mean()
            head_res[h] = dict(aucs=A_h, mean_auc=float(np.nanmean(A_h)), P=P_h, routed=float(acc))
            lines.append(f"| {h} | " + " | ".join(f"{a:.3f}" for a in A_h) +
                         f" | {head_res[h]['mean_auc']:.3f} | {acc:.3f} | " +
                         " | ".join(f"{C[r].mean():.3f}" for r in routed) + f" | {stack.max(axis=0).mean():.3f} |")
            print(f"head {h:<7} AUC {[round(a, 3) for a in A_h]}  routed {acc:.3f}")
        except Exception as e:                      # e.g. xgboost not installed
            lines.append(f"| {h} | " + " | ".join("-" for _ in routed) + " | - | - | " +
                         " | ".join("-" for _ in routed) + f" | ({type(e).__name__}) |")
            print(f"head {h}: skipped ({type(e).__name__}: {e})")
    if config.ROUTER_HEAD == "auto":
        head = max(head_res, key=lambda h: (round(head_res[h]["routed"], 4), head_res[h]["mean_auc"]))
    else:
        head = config.ROUTER_HEAD
        if head not in head_res:
            sys.exit(f"router head {head} did not train")
    lines += ["", f"**Head used: `{head}`** ({'best OOF routed accuracy, ties by mean AUC' if config.ROUTER_HEAD == 'auto' else 'fixed in config'})."]
    print(f"router head: {head}")
    P = head_res[head]["P"]
    pick = P.argmax(axis=0)

    # where do the picks go, per dataset (OOF)?
    lines += ["", "## Router picks per dataset (OOF, train)", "",
              "| dataset | n | " + " | ".join(f"-> {r}" for r in routed) + " | routed acc | " +
              " | ".join(f"always-{r}" for r in routed) + " |", "|---|---:|" + "---:|" * (2 * len(routed) + 1)]
    for ds in config.DATASETS:
        m = ds_of == ds
        if not m.any():
            continue
        lines.append(f"| {ds} | {int(m.sum())} | " + " | ".join(f"{(pick[m] == j).mean():.0%}" for j in range(len(routed))) +
                     f" | {stack[pick[m], np.where(m)[0]].mean():.3f} | " +
                     " | ".join(f"{C[r][m].mean():.3f}" for r in routed) + " |")

    # fit the chosen head on all train rows, per routed model
    router = {"routed": routed, "escalation": esc_role, "pca_dims": config.PCA_DIMS, "head": head,
              "features": "pca32", "models": {}}
    for j, r in enumerate(routed):
        pca = PCA(config.PCA_DIMS, random_state=0).fit(E)
        X = pca.transform(E)
        sc = StandardScaler().fit(X)
        clf = make_head(head, 0, best_C[r]).fit(sc.transform(X), C[r])
        router["models"][r] = {"pca": pca, "scaler": sc, "clf": clf, "C": best_C[r],
                               "oof_auc": head_res[head]["aucs"][j]}
    with open(config.ROUTER_DIR / "router.pkl", "wb") as f:
        pickle.dump(router, f)

    # ---- "whether": does each routed model's confidence find its own errors?
    lines += ["", "## Confidence as a failure signal (AUC of 1 - confidence for 'model is wrong')", "",
              "| model | all train | on its routed questions |", "|---|---:|---:|"]
    for j, r in enumerate(routed):
        y = 1 - C[r]
        a_all = roc_auc_score(y, -CONF[r]) if 0 < y.mean() < 1 else np.nan
        m = pick == j
        a_rt = roc_auc_score(y[m], -CONF[r][m]) if m.sum() > 10 and 0 < y[m].mean() < 1 else np.nan
        lines.append(f"| {r} | {a_all:.3f} | {a_rt:.3f} |")

    # ---- tau per route (on the OOF-routed questions), one global tau, and the plain cascade's tau
    tau_per_route, sweeps = {}, {}
    lines += ["", f"## tau per route (train, OOF picks): routed model answers; confidence < tau -> {esc_role}", ""]
    for j, r in enumerate(routed):
        m = pick == j
        sweep, t = choose_tau(CONF[r][m], C[r][m], C[esc_role][m])
        tau_per_route[r], sweeps[r] = t, sweep
        lines += [f"### route -> {r}  (n = {int(m.sum())}, tau* = {t})", "",
                  "| tau | escalated | acc |", "|---:|---:|---:|"]
        lines += [f"| {s['tau']:.2f} | {s['escalated']:.1%} | {s['acc']:.3f} |" for s in sweep]
        lines.append("")
    # global tau: one threshold shared by the routes
    conf_pick = CONF[routed[0]].copy()
    c_pick = stack[pick, np.arange(n)]
    for j, r in enumerate(routed):
        conf_pick[pick == j] = CONF[r][pick == j]
    g_sweep, tau_global = choose_tau(conf_pick, c_pick, C[esc_role])
    for r in routed:                       # a never-picked route falls back to the global tau
        if tau_per_route[r] is None:
            tau_per_route[r] = tau_global
            lines.append(f"(route -> {r} was never picked on train; its tau falls back to the global tau)")
    # plain cascade: generalist alone first, no router
    p_sweep, tau_cascade = choose_tau(CONF[config.BASE], C[config.BASE], C[esc_role])
    lines += [f"### one global tau for route-then-cascade: tau* = {tau_global}", "",
              "| tau | escalated | acc |", "|---:|---:|---:|"]
    lines += [f"| {s['tau']:.2f} | {s['escalated']:.1%} | {s['acc']:.3f} |" for s in g_sweep]
    lines += ["", f"### plain cascade {config.BASE} -> {esc_role} (no router): tau* = {tau_cascade}", "",
              "| tau | escalated | acc |", "|---:|---:|---:|"]
    lines += [f"| {s['tau']:.2f} | {s['escalated']:.1%} | {s['acc']:.3f} |" for s in p_sweep]

    # summary of the train-split decision rule
    acc_rc = sum(np.where(CONF[r][pick == j] < tau_per_route[r], C[esc_role][pick == j], C[r][pick == j]).sum()
                 for j, r in enumerate(routed)) / n
    lines += ["", "## Train-split summary (OOF)", "",
              f"- route only (no escalation): {c_pick.mean():.3f}",
              f"- route then cascade, per-route tau: {acc_rc:.3f}",
              f"- plain cascade {config.BASE} -> {esc_role} at tau {tau_cascade}: "
              f"{max(p_sweep, key=lambda s: s['tau'] == tau_cascade)['acc']:.3f}",
              f"- always-{esc_role}: {C[esc_role].mean():.3f}",
              f"- oracle over routed tier: {stack.max(axis=0).mean():.3f}; oracle over all: "
              f"{np.maximum(stack.max(axis=0), C[esc_role]).mean():.3f}"]

    thresholds = {"tau_star": tau_global, "tau_per_route": tau_per_route, "tau_cascade": tau_cascade,
                  "routed": routed, "escalation": esc_role, "head": head,
                  "sweeps": {"per_route": sweeps, "global": g_sweep, "cascade": p_sweep},
                  "head_ablation": {h: {"aucs": v["aucs"], "mean_auc": v["mean_auc"], "routed": v["routed"]}
                                    for h, v in head_res.items()}}
    (config.ROUTER_DIR / "thresholds.json").write_text(json.dumps(thresholds, indent=2))
    (config.RESULTS_DIR / "router_train.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\ntau per route = {tau_per_route}   global tau* = {tau_global}   cascade tau = {tau_cascade}")
    print("wrote router/router.pkl, router/thresholds.json, results/router_train.md")


if __name__ == "__main__":
    main()
