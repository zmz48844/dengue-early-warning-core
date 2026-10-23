from __future__ import annotations

import os
import shutil
import sys
import time
import traceback

from main import engine as f
from main.engine import e, np, pd, Path
from common import paths, data_builder

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def status(state, **details):
    path = f.OUT / "run_status.json"
    old = f.read_json(path) if f.exists(path) else {}
    f.dump(dict(old, status=state, updated_at=f.stamp(), pid=os.getpid(),
                test_opened_by_this_run=f.exists(f.OUT / "test_access_audit.json"), **details), path)


def assert_pre_test():
    assert not f.exists(f.OUT / "test_access_audit.json"), "This run has accessed Test: selection/refit is forbidden; use locked systems only"


def save_search_rows(rows, path):
    flat = []
    for row in rows:
        value = dict(candidate_id=row["candidate_id"], arm=row.get("arm"),
                     selected_gate=row.get("selected_gate"), feasible=row.get("feasible"),
                     total_shortfall=row.get("total_shortfall"),
                     summed_policy_false_weeks=row.get("summed_policy_false_weeks"),
                     **row.get("countdown_potential", {}))
        for name, metric in row.get("system", {}).items():
            value.update({name+"_"+k: v for k, v in metric.items()})
        value["params"] = __import__("json").dumps(row["params"], sort_keys=True)
        flat.append(value)
    f.save_csv(pd.DataFrame(flat), path)


def select_stage1(d, candidates, ctx):
    assert_pre_test()
    status("RUNNING", phase="Stage1 raw-OOF selection")
    y, valid = e.label_and_valid(d["meta"])
    rows, raw_by_id, audits = [], {}, []
    for candidate in candidates:
        status("RUNNING", phase="Stage1 raw-OOF selection", arm=None,
               completed_candidates=len(rows), total_candidates=len(candidates), current_candidate=candidate["id"])
        native, audit = f.raw_oof(d, candidate, 1, None, ctx)
        row = dict(candidate_id=candidate["id"], params=candidate["params"],
                   average_precision_raw=float(e.average_precision_score(y[valid], native[valid])),
                   brier_raw=float(e.brier_score_loss(y[valid], native[valid])))
        rows.append(row)
        raw_by_id[candidate["id"]] = native
        audits.append(audit)
        f.dump(rows, f.OUT / "stage1_search_progress.json")
        f.log(f"Stage1 {candidate['id']:02d}/{len(candidates)} raw OOF AP={row['average_precision_raw']:.5f}; cached={audit['cached_blocks']}/3, fitted={audit['fitted_blocks']}")
    status("RUNNING", phase="Stage1 raw-OOF selection complete", arm=None,
           completed_candidates=len(rows), total_candidates=len(candidates), current_candidate=None)
    best = min(rows, key=lambda row: (-row["average_precision_raw"], row["brier_raw"], row["candidate_id"]))
    raw = raw_by_id[best["candidate_id"]]
    calibrator = e.fit_probability_calibrator(d["meta"], raw)
    probability = e.calibrated_probability(calibrator, raw)
    best.update(calibration_source="pooled raw development OOF", gate_not_selected_at_this_step=True,
                probability_metrics=e.probability_metrics(d["meta"], raw, probability))
    f.dump(best, f.OUT / "selected_stage1.json")
    f.dump(rows, f.OUT / "stage1_search.json")
    f.save_csv(pd.DataFrame([dict(candidate_id=r["candidate_id"], average_precision_raw=r["average_precision_raw"],
                                brier_raw=r["brier_raw"], **r["params"]) for r in rows]), f.OUT / "stage1_search.csv")
    f.save_joblib(dict(raw=raw, probability=probability, calibrator=calibrator), f.OUT / "selected_stage1_oof.joblib")
    f.dump(audits, f.OUT / "stage1_cache_audit.json")
    f.log(f"Stage1 fixed candidate={best['candidate_id']}; Gate awaits full-system selection")
    return best, raw, probability, calibrator


def select_stage2(d, candidates, arm, probability, ctx):
    assert_pre_test()
    status("RUNNING", phase="Stage2 and Gate development selection", arm=arm)
    evaluator = f.GateEvaluator(d["meta"], probability)
    rows, all_gates, audits = [], [], []
    best_key, best_values = None, None
    for candidate in candidates:
        status("RUNNING", phase="Stage2 and Gate development selection", arm=arm,
               completed_candidates=len(rows), total_candidates=len(candidates), current_candidate=candidate["id"])
        native, audit = f.raw_oof(d, candidate, 2, arm, ctx)
        calibration = e.fit_interval_calibrator(d["meta"], native)
        intervals = e.apply_interval_calibrator(native, calibration)
        gate, gate_rows = evaluator.evaluate(native)
        row = dict(candidate_id=candidate["id"], arm=arm, params=candidate["params"],
                   selected_gate=gate["gate"], feasible=gate["feasible"],
                   total_shortfall=gate["total_shortfall"], shortfalls=gate["shortfalls"],
                   system=gate["system"], summed_policy_false_weeks=gate["summed_policy_false_weeks"],
                   feasible_gate_count=sum(g["feasible"] for g in gate_rows),
                   countdown_potential=f.countdown_metrics(d["meta"], native, intervals),
                   interval_calibration=calibration)
        rows.append(row)
        audits.append(audit)
        all_gates.extend(dict(candidate_id=candidate["id"], **g) for g in gate_rows)
        key = f.candidate_key(row)
        if best_key is None or key < best_key:
            best_key, best_values = key, (native, intervals, calibration)
        save_search_rows(rows, f.OUT / f"stage2_search_{arm}.csv")
        f.dump(rows, f.OUT / f"stage2_search_progress_{arm}.json")
        if candidate["id"] == 1 or candidate["id"] % 4 == 0:
            f.log(f"Stage2 {arm} {candidate['id']:02d}/{len(candidates)} feasible={sum(r['feasible'] for r in rows)}; best shortfall={best_key[1]:.4f}, IS321={best_key[2]:.4f}, MAE321={best_key[3]:.4f}")
    status("RUNNING", phase="Stage2 and Gate development selection complete", arm=arm,
           completed_candidates=len(rows), total_candidates=len(candidates), current_candidate=None)
    selection = f.select_record(rows)
    native, intervals, calibrator = best_values
    f.dump(selection, f.OUT / f"selection_{arm}.json")
    f.dump(rows, f.OUT / f"stage2_search_{arm}.json")
    f.dump(all_gates, f.OUT / f"gate_search_{arm}.json")
    f.dump(audits, f.OUT / f"stage2_cache_audit_{arm}.json")
    flat_gates = []
    for g in all_gates:
        value = {k: g[k] for k in ["candidate_id", "gate", "feasible", "total_shortfall", "summed_policy_false_weeks"]}
        value.update({name+"_"+metric: val for name, metrics in g["system"].items() for metric, val in metrics.items()})
        flat_gates.append(value)
    f.save_csv(pd.DataFrame(flat_gates), f.OUT / f"gate_search_{arm}.csv")
    f.save_joblib(dict(native=native, intervals=intervals, calibrator=calibrator), f.OUT / f"selected_stage2_oof_{arm}.joblib")
    row = selection["prediction_candidate"]
    f.log(f"Development {arm}: {selection['status']}; c{row['candidate_id']} Gate={row['selected_gate']:.4f}; " +
          str({name: row['system'][name]['event_recall'] for name in f.CONSTRAINTS}) +
          f"; blue specificity={row['system']['blue']['normal_week_specificity']:.4f}")
    return selection, native, intervals, calibrator


def lock_decisions(d, stage1, probability_calibrator, selections, calibrators, ctx):
    assert_pre_test()
    blueprints = {}
    for arm in e.ARMS:
        selection = selections[arm]
        row = selection["prediction_candidate"]
        blueprint = dict(protocol_fingerprint=ctx["fingerprint"], stage1_params=stage1["params"],
                         stage1_candidate=stage1["candidate_id"], stage2_params=row["params"],
                         candidate_id=row["candidate_id"], formal_candidate_id=selection["selected_candidate_id"],
                         diagnostic_candidate_id=selection["diagnostic_candidate_id"], arm=arm,
                         gate=selection["selected_gate"], status=selection["status"], feasible=row["feasible"],
                         probability_calibrator=dict(x=probability_calibrator.X_thresholds_.tolist(), y=probability_calibrator.y_thresholds_.tolist()),
                         interval_calibrator=calibrators[arm], training_counts=e.data_counts(d["meta"]),
                         refit_seeds={"stage1": e.SEED+901, "stage2": e.SEED+902})
        signature = e.signature(blueprint)
        path = f.OUT / "final" / arm / "decision_lock.json"
        if f.exists(path):
            assert f.read_json(path)["signature"] == signature, "A locked decision changed"
        else:
            f.dump(dict(blueprint, signature=signature, locked_at=f.stamp(), test_loaded_by_this_run=False), path)
        blueprints[arm] = dict(blueprint, signature=signature)
    eligible = [a for a in e.ARMS if selections[a]["prediction_candidate"]["feasible"]]
    primary = min(eligible, key=lambda a: (f.candidate_key(selections[a]["prediction_candidate"]), e.ARMS.index(a))) if eligible else None
    record = dict(primary_arm=primary, status="QUALIFIED" if primary else "NO_FEASIBLE_PRIMARY",
                  selected_on="development OOF only, before refit and Test", arms_reported=list(e.ARMS),
                  decision_signatures={a: blueprints[a]["signature"] for a in e.ARMS})
    record["signature"] = e.signature(record)
    path = f.OUT / "primary_arm_lock.json"
    if f.exists(path):
        assert f.read_json(path)["signature"] == record["signature"]
    else:
        f.dump(dict(record, locked_at=f.stamp()), path)
    f.log(f"Both decision/calibration/Gate specifications locked; formal primary arm={primary}")
    return blueprints


def refit(d, blueprints, probability_calibrator, interval_calibrators, ctx):
    assert_pre_test()
    status("RUNNING", phase="Full-development refit with locked decisions")
    all_rows = np.arange(len(d["meta"]))
    s1_signature = e.signature(dict(protocol=ctx["fingerprint"], stage1=blueprints[e.ARMS[0]]["stage1_params"], seed=e.SEED+901))
    s1_path = f.OUT / "final" / "refit_stage1.joblib"
    if f.exists(s1_path):
        saved = f.load_joblib(s1_path)
        assert saved["signature"] == s1_signature
        fitted1 = saved["fitted"]
    else:
        f.log("Refitting shared Stage1 on all development city-years")
        _, fitted1 = e.fit_stage1(d, all_rows, [], blueprints[e.ARMS[0]]["stage1_params"], e.SEED+901)
        f.save_joblib(dict(signature=s1_signature, fitted=fitted1), s1_path)
    for arm in e.ARMS:
        specification = blueprints[arm]
        folder = f.OUT / "final" / arm
        model_path = folder / "system.joblib"
        if f.exists(model_path):
            bundle = f.load_joblib(model_path)
            assert bundle["signature"] == specification["signature"]
        else:
            f.log(f"Refitting {arm} Stage2 on all {int(e.exact_mask(d['ttw']).sum())} true TTW rows")
            _, fitted2 = e.fit_stage2(d, all_rows, [], specification["stage2_params"], arm, e.SEED+902)
            bundle = dict(signature=specification["signature"], stage1=fitted1, stage2=fitted2,
                          probability_calibrator=probability_calibrator,
                          interval_calibrator=interval_calibrators[arm], gate=specification["gate"],
                          features=d["features"], arm=arm, feasible=specification["feasible"],
                          selection_status=specification["status"], training_counts=e.data_counts(d["meta"]))
            f.save_joblib(bundle, model_path)
        lock = dict(specification, model_sha256=f.file_hash(model_path), model_path=model_path.relative_to(f.OUT).as_posix(),
                    calibration_reestimated_after_refit=False, fitted_in_sample_Gate_selection=False)
        lock_path = folder / "system_lock.json"
        if f.exists(lock_path):
            old = f.read_json(lock_path)
            assert old["signature"] == lock["signature"] and old["model_sha256"] == lock["model_sha256"]
        else:
            f.dump(dict(lock, locked_at=f.stamp(), test_opened_by_this_run=False), lock_path)
        del bundle


def predict(d, bundle):
    assert d["features"] == bundle["features"]
    x1 = bundle["stage1"]["scaler"].transform(d["X"][:, 0, :]).astype(np.float32)[:, None, :]
    raw1 = bundle["stage1"]["model"].predict_12w_raw(x1)["p_12w_raw"]
    probability = e.calibrated_probability(bundle["probability_calibrator"], raw1)
    model2 = bundle["stage2"]
    x2 = model2["scaler"].transform(d["X"][:, 0, :]).astype(np.float32)
    native = np.column_stack([e.inverse_target(model.predict(x2), model2["arm"], model2["tau"]) for model in model2["models"]])
    intervals = e.apply_interval_calibrator(native, bundle["interval_calibrator"])
    return f.build_frame(d, raw1, probability, native, intervals, bundle["gate"], bundle["arm"], bundle["feasible"], bundle["selection_status"])


def evaluate_test():
    lock_paths = [f.OUT / "final" / arm / "system_lock.json" for arm in e.ARMS]
    lock_paths.append(f.OUT / "primary_arm_lock.json")
    assert all(f.exists(p) for p in lock_paths), "Both systems and primary arm must be locked before Test access"
    hashes = {p.relative_to(f.OUT).as_posix(): f.file_hash(p) for p in lock_paths}
    for arm in e.ARMS:
        lock = f.read_json(f.OUT / "final" / arm / "system_lock.json")
        assert f.file_hash(f.OUT / "final" / arm / "system.joblib") == lock["model_sha256"]
    access = f.OUT / "test_access_audit.json"
    if f.exists(access):
        assert f.read_json(access)["pre_existing_lock_hashes"] == hashes, "Locked system changed after Test access"
    else:
        f.dump(dict(first_test_access_in_this_run=f.stamp(), pre_existing_lock_hashes=hashes,
                    earlier_project_Test_access=True, selection_after_this_access_forbidden=True), access)
    status("RUNNING", phase="Locked temporal retrospective evaluation")
    source = paths.INPUT_CACHE_DIR / "test_47feature_cache.joblib"
    cached = data_builder.load_test()
    test = {k: cached[k] for k in ["X", "ttw", "meta", "features"]}
    test["meta"] = test["meta"].drop(columns=["development_role"], errors="ignore").copy()
    assert test["meta"].year.between(2019, 2023).all()
    assert not test["meta"].duplicated(["province", "city", "year", "sample_idx"]).any()
    assert len(test["features"]) == 47 and f.PROTOCOL["removed_feature"] not in test["features"]
    assert np.allclose(test["ttw"], test["meta"].raw_ttw, equal_nan=True)
    counts = e.data_counts(test["meta"])
    assert all(counts[k] == v for k, v in f.PROTOCOL["expected_test"].items())
    identity = dict(source=source.relative_to(paths.ROOT).as_posix(), counts=counts, features=test["features"],
                    X_hash=e.array_digest(test["X"]), ttw_hash=e.array_digest(test["ttw"]),
                    metadata_hash=e.array_digest(pd.util.hash_pandas_object(test["meta"], index=False).to_numpy()))
    identity_path = f.OUT / "test_identity.json"
    if f.exists(identity_path):
        assert f.read_json(identity_path) == identity
    else:
        f.dump(identity, identity_path)
    for arm in e.ARMS:
        folder = f.OUT / "test" / arm
        bundle = f.load_joblib(f.OUT / "final" / arm / "system.joblib")
        pred_path = folder / "predictions.joblib"
        if f.exists(pred_path):
            saved = f.load_joblib(pred_path)
            assert saved["model_signature"] == bundle["signature"]
            frame = saved["frame"]
        else:
            frame = predict(test, bundle)
            f.save_joblib(dict(frame=frame, model_signature=bundle["signature"]), pred_path)
            f.save_csv(frame, folder / "predictions.csv")
        result = f.evaluate_frame(frame, folder, "2019_2023_temporal_retrospective")
        f.log(f"Test {arm}: constraints_met={result['evaluation_constraints_met']}; " +
              str({name: result['system'][name]['event_recall'] for name in f.CONSTRAINTS}) +
              f"; blue specificity={result['system']['blue']['normal_week_specificity']:.4f}")
        del bundle


def archive_sources():
    folder = f.OUT / "source_snapshot"
    folder.mkdir(parents=True, exist_ok=True)
    for relative, digest in f.source_hashes().items():
        source = f.ROOT / relative
        target = folder / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if f.exists(target):
            assert f.file_hash(target) == digest, "Frozen source changed"
        else:
            shutil.copy2(f.io_path(source), f.io_path(target))


def verify_run_sources():
    lock = f.read_json(f.OUT / "protocol_lock.json")
    versions = {"numpy_version": np.__version__, "pandas_version": pd.__version__,
                "sklearn_version": f.sklearn.__version__, "scipy_version": f.scipy.__version__}
    for key, value in versions.items():
        if key in lock and lock[key] != value:
            raise RuntimeError("Numerical environment changed since locking: " + key)
    if lock["source_hashes"] != f.source_hashes():
        raise RuntimeError("Scientific source changed since this run was locked; use a new run-id")
    if lock["data_config_sha256"] != f.file_hash(paths.DATA_CONFIG_PATH):
        raise RuntimeError("Data configuration changed since locking")
    return lock


def finish_audit(lock, began):
    acceptance = {arm: f.read_json(f.OUT / "test" / arm / "metrics.json")["evaluation_constraints_met"] for arm in e.ARMS}
    audits = []
    for name in ["stage1_cache_audit.json", *[f"stage2_cache_audit_{a}.json" for a in e.ARMS]]:
        audits.extend(f.read_json(f.OUT / name))
    candidate_counts = dict(stage1=len(lock["stage1_candidates"]), stage2={a:len(v) for a,v in lock["stage2_candidates"].items()})
    total = candidate_counts["stage1"]+sum(candidate_counts["stage2"].values())
    assert sum(a["fitted_blocks"]+a["cached_blocks"] for a in audits) == 3*total
    f.dump(dict(completed_at=f.stamp(), protocol_fingerprint=lock["fingerprint"],
                source_hashes=f.source_hashes(), candidate_oof_blocks=3*total,
                fitted_blocks_in_last_selection=sum(a["fitted_blocks"] for a in audits),
                cached_blocks_in_last_selection=sum(a["cached_blocks"] for a in audits),
                no_external_experiment_dependency=True, candidate_counts=candidate_counts,
                recall_constraints=f.CONSTRAINTS, minimum_blue_specificity=f.MIN_BLUE_SPEC,
                temporal_acceptance_by_arm=acceptance, primary_arm=f.read_json(f.OUT / "primary_arm_lock.json")["primary_arm"],
                no_post_test_selection=True, computational_completion_is_not_performance_success=True),
           f.OUT / "final_integrity_audit.json")
    status("COMPLETED", phase="All outputs complete", finished_at=f.stamp(), arm=None,
           completed_candidates=total, total_candidates=total, current_candidate=None,
           candidate_counts=candidate_counts, this_execution_seconds=time.monotonic()-began,
           temporal_acceptance_by_arm=acceptance)


def execute(mode="train"):

    if mode not in {"prepare", "train", "test", "all"}:
        raise ValueError("Unknown execution mode")
    paths.require_configured()
    f.OUT.mkdir(parents=True, exist_ok=True)
    lease = f.OUT / "running.lock"
    try:
        descriptor = os.open(f.io_path(lease), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError("This run has a running.lock. Confirm its process has exited before manually removing only that lease; never delete result files.")
    os.write(descriptor, __import__("json").dumps(dict(pid=os.getpid(), created_at=f.stamp(), script=str(Path(__file__).resolve()))).encode("utf-8"))
    os.close(descriptor)
    began = time.monotonic()
    try:
        if mode == "test":
            lock = verify_run_sources()
            evaluate_test()
            finish_audit(lock, began)
            return
        if mode in {"prepare", "train"}:
            assert_pre_test()
        d, s1, s2, ctx = f.prepare()
        archive_sources()
        if mode == "prepare":
            status("PREPARED", phase="Protocol frozen; no Test read", protocol_fingerprint=ctx["fingerprint"])
            return
        if not f.exists(f.OUT / "test_access_audit.json"):
            stage1, raw1, probability, probability_calibrator = select_stage1(d, s1, ctx)
            selections, quantiles, intervals, interval_calibrators = {}, {}, {}, {}
            for arm in e.ARMS:
                selections[arm], quantiles[arm], intervals[arm], interval_calibrators[arm] = select_stage2(d, s2[arm], arm, probability, ctx)
            blueprints = lock_decisions(d, stage1, probability_calibrator, selections, interval_calibrators, ctx)
            for arm in e.ARMS:
                selection = selections[arm]
                frame = f.build_frame(d, raw1, probability, quantiles[arm], intervals[arm], selection["selected_gate"],
                                      arm, selection["prediction_candidate"]["feasible"], selection["status"])
                frame["development_fold"] = ctx["folds"]+1
                folder = f.OUT / "development_oof" / arm
                f.save_joblib(dict(frame=frame, model_signature=blueprints[arm]["signature"]), folder / "predictions.joblib")
                f.save_csv(frame, folder / "predictions.csv")
                f.evaluate_frame(frame, folder, "development_OOF_tuning_not_independent_validation")
                for fold in range(1, 4):
                    f.evaluate_frame(frame.loc[frame.development_fold == fold].reset_index(drop=True),
                                     folder / f"fold_{fold}", f"development_tuning_fold_{fold}", bootstrap=False)
            refit(d, blueprints, probability_calibrator, interval_calibrators, ctx)
        else:
            f.log("Test already accessed: no selection/refit; reuse locked systems only")
        if mode == "train":
            status("TRAINED", phase="Both systems locked and refitted; Test not accessed", protocol_fingerprint=ctx["fingerprint"])
            return
        evaluate_test()
        finish_audit(f.read_json(f.OUT / "protocol_lock.json"), began)
    except BaseException as error:
        status("INTERRUPTED" if isinstance(error, KeyboardInterrupt) else "FAILED", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        os.unlink(f.io_path(lease))
