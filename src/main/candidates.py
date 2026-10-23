from __future__ import annotations

from common import modeling as e
from common.modeling import np, qmc


STAGE1_COUNT = 96
STAGE2_COUNT = 144
TAU_BLOCK_SIZE = 48
TAU_VALUES = (2., 3., 4., 5., 6., 8.)
CANDIDATE_METADATA = {
    "method": "fixed scrambled Sobol prefix with unchanged generic first anchor",
    "stage1_count": STAGE1_COUNT,
    "stage2_count_per_arm": STAGE2_COUNT,
    "stage1_sobol": {"dimensions": 4, "seed": e.SEED, "power": 7, "take_first": STAGE1_COUNT},
    "stage2_sobol": {"dimensions": 7, "seed": e.SEED+1, "power": 8, "take_first": STAGE2_COUNT},
    "exponential_tau": {
        "values_weeks": list(TAU_VALUES),
        "seed": e.SEED+2,
        "blocks": 3,
        "block_size": TAU_BLOCK_SIZE,
        "copies_per_tau_per_block": 8,
        "rule": "one seeded RNG sequentially shuffles three separate balanced 48-item blocks; concatenate without reshuffling",
    },
    "structure_matching": "both TTW arms use the same 144 non-tau hyperparameter structures",
    "prefix_requirement": "fixed candidate order and sequential 48-item exponential-tau blocks",
    "ranges": "unchanged from the original predeclared Sobol space; no result-dependent anchors or range changes",
}


def candidates():

    points = qmc.Sobol(4, scramble=True, seed=e.SEED).random_base2(7)[:STAGE1_COUNT]
    stage1 = []
    for i, z in enumerate(points):
        stage1.append({"id": i+1, "params": {
            "rf_n_estimators": 250,
            "rf_max_depth": int(round(8+28*z[0])),
            "rf_min_samples_leaf": int(round(5+115*z[1])),
            "rf_max_features": float(np.exp(np.log(.04)+np.log(15)*z[2])),
            "rf_max_samples": float(.55+.45*z[3]),
        }})
    stage1[0]["params"].update(
        rf_max_depth=16, rf_min_samples_leaf=20, rf_max_features=.15, rf_max_samples=.8)

    points = qmc.Sobol(7, scramble=True, seed=e.SEED+1).random_base2(8)[:STAGE2_COUNT]
    rng = np.random.default_rng(e.SEED+2)
    tau_blocks = []

    for _ in range((STAGE2_COUNT+TAU_BLOCK_SIZE-1)//TAU_BLOCK_SIZE):
        block = np.tile(TAU_VALUES, 8)
        rng.shuffle(block)
        tau_blocks.append(block)
    taus = np.concatenate(tau_blocks)[:STAGE2_COUNT]

    stage2 = {arm: [] for arm in e.ARMS}
    for i, z in enumerate(points):
        params = {
            "learning_rate": float(np.exp(np.log(.01)+np.log(18)*z[0])),
            "max_depth": int(round(2+6*z[1])),
            "max_leaf_nodes": int(round(7+56*z[2])),
            "min_samples_leaf": int(round(2+98*z[3])),
            "l2_regularization": float(np.exp(np.log(.001)+np.log(30000)*z[4])),
            "max_bins": [63, 127, 255][min(int(z[5]*3), 2)],
            "max_iter": 250+50*int(round(z[6]*23)),
        }
        if i == 0:
            params = dict(learning_rate=.05, max_depth=4, max_leaf_nodes=31,
                          min_samples_leaf=10, l2_regularization=1., max_bins=127, max_iter=600)
        for arm in e.ARMS:
            stage2[arm].append({
                "id": i+1, "arm": arm,
                "params": dict(params, tau_weeks=float(taus[i]))
                if arm == "exponential" else params.copy(),
            })

    assert len(stage1) == STAGE1_COUNT
    assert all(len(stage2[arm]) == STAGE2_COUNT for arm in e.ARMS)
    for exp, raw in zip(stage2["exponential"], stage2["direct_ttw"]):
        assert {k: v for k, v in exp["params"].items() if k != "tau_weeks"} == raw["params"]
    return stage1, stage2
