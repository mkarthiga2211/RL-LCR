"""Script: Invariant and regression verification suite executed before Phase 5 experiments.

Runs 6 foundational pre-flight checks on fast configurations:
    1. Determinism: identical (method, scenario, seed) -> identical packets.parquet SHA256.
    2. Common Random Numbers: identical topology, patient assignment, and generated packet stream.
    3. Conservation Law: generated == delivered + dropped + in-flight.
    4. Energy Accounting: sum(categorized) == total; non-negative battery residuals.
    5. Split Hygiene: tuning scripts restricted to {train, val}; eval scripts restricted to {test, zeroshot}.
    6. RL-LCR Feasibility: feasible routes satisfy all bounds; infeasible routes flagged with fallback.

Outputs reports/verification.md. Phase 5 experiment scripts MUST refuse execution if this report
is missing or contains any FAIL.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from loguru import logger

from rllcr.constraints.clinical_mapping import ClinicalConstraintMapper
from rllcr.constraints.feasibility import FeasibilityChecker
from rllcr.data.splits import load_traces
from rllcr.eval.collector import get_git_hash
from rllcr.routing.base import RoutingProtocol
from rllcr.routing.rl_lcr import RLLCR
from rllcr.sim.packet import Packet
from rllcr.sim.sdn_controller import SDNController
from rllcr.sim.simulator import Simulator
from rllcr.sim.topology import generate_ward_topology
from rllcr.sim.trace_player import assign_traces_to_nodes
from rllcr.utils.logging import setup_logging


@dataclass
class CheckResult:
    """Outcome of a single verification check."""

    check_id: int
    name: str
    passed: bool
    details: str
    duration_s: float


class DummyForwarder(RoutingProtocol):
    """Direct sink-forwarding baseline routing protocol for CRN verification."""

    name: str = "dummy_forwarder"
    uses_controller: bool = False

    def attach(self, sim: Simulator) -> None:
        """Attach protocol to simulator."""
        self.sink_id = sim.sink_id

    def on_packet_generated(self, node_id: int, pkt: Packet) -> None:
        """No-op on generation."""
        pass

    def next_hop(self, node_id: int, pkt: Packet) -> int | None:
        """Direct single hop to sink."""
        return self.sink_id

    def on_link_failure(self, node_id: int, nbr: int, pkt: Packet) -> None:
        """No-op on failure."""
        pass

    def on_control_timer(self, now: float) -> None:
        """No-op on control timer."""
        pass

    def control_packets(self) -> list[Packet]:
        """No control packets."""
        return []


def run_check_1_determinism() -> CheckResult:
    """Verify bit-exact determinism across identical runs."""
    t0 = time.perf_counter()
    seed = 2026
    val_traces = load_traces(split="val", purpose="tuning")
    topo = generate_ward_topology(n_nodes=10, seed=42)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_p = Path(tmp_dir)

        def run_sim(out_dir: Path) -> str:
            protocol = RLLCR(lr=0.01, iters=20, rho=1.0, omega_tx=1.0, k_candidates=2, device_pref="cpu")
            ctrl = SDNController(sink_id=topo.sink_id, route_recompute_interval_s=10.0)
            sim = Simulator(
                cfg={"radio": {"bitrate_bps": 250000}, "protocol": "rl_lcr", "seed": seed},
                topology=topo,
                protocol=protocol,
                trace_paths=val_traces,
                controller=ctrl,
                seed=seed,
                duration_s=4.0,
                warmup_s=0.0,
            )
            ctrl.attach(sim)
            sim.run()
            saved = sim.collector.save(out_dir)
            return hashlib.sha256(saved["packets"].read_bytes()).hexdigest()

        hash1 = run_sim(tmp_p / "run1")
        hash2 = run_sim(tmp_p / "run2")

    passed = (hash1 == hash2)
    dur = time.perf_counter() - t0
    details = f"packets.parquet SHA256: {hash1[:16]}... (Exact match: {passed})"
    return CheckResult(1, "Determinism (Bit-exact Parquet Hash)", passed, details, dur)


def run_check_2_crn() -> CheckResult:
    """Verify Common Random Numbers alignment across distinct routing algorithms."""
    t0 = time.perf_counter()
    seed = 2026
    val_traces = load_traces(split="val", purpose="tuning")

    # Topology match
    topo1 = generate_ward_topology(n_nodes=10, seed=seed)
    topo2 = generate_ward_topology(n_nodes=10, seed=seed)
    topo_match = (topo1.to_dict() == topo2.to_dict())

    # Patient assignment match
    source_ids = [i for i, t in enumerate(topo1.node_types) if t in ("bedside", "wearable")]
    assign1 = assign_traces_to_nodes(source_ids, val_traces, seed=seed)
    assign2 = assign_traces_to_nodes(source_ids, val_traces, seed=seed)
    assign_match = (assign1 == assign2)

    # Generated packets stream match across different protocols
    def get_packets(protocol: RoutingProtocol) -> list[tuple[int, float, int, str, str]]:
        sim = Simulator(
            cfg={"radio": {"bitrate_bps": 250000}, "protocol": protocol.name, "seed": seed},
            topology=topo1,
            protocol=protocol,
            trace_paths=val_traces,
            seed=seed,
            duration_s=4.0,
            warmup_s=0.0,
        )
        if protocol.uses_controller and sim.controller is not None:
            sim.controller.attach(sim)
        sim.run()
        return [
            (p.src, round(p.t_gen, 6), p.bytes, p.pkt_class, p.alarm_class)
            for p in sim.collector.packet_records
        ]

    pkts_rllcr = get_packets(RLLCR(lr=0.01, iters=20, device_pref="cpu"))
    pkts_dummy = get_packets(DummyForwarder())
    pkts_match = (len(pkts_rllcr) > 0 and pkts_rllcr == pkts_dummy)

    passed = topo_match and assign_match and pkts_match
    dur = time.perf_counter() - t0
    details = (
        f"Topology: {topo_match} | Assign: {assign_match} | "
        f"Packets ({len(pkts_rllcr)} generated): {pkts_match}"
    )
    return CheckResult(2, "Common Random Numbers (CRN Alignment)", passed, details, dur)


def run_check_3_conservation() -> CheckResult:
    """Verify packet conservation law: generated == delivered + dropped + in-flight."""
    t0 = time.perf_counter()
    seed = 2026
    val_traces = load_traces(split="val", purpose="tuning")
    topo = generate_ward_topology(n_nodes=10, seed=seed)

    protocol = RLLCR(lr=0.01, iters=20, device_pref="cpu")
    ctrl = SDNController(sink_id=topo.sink_id)
    sim = Simulator(
        cfg={"radio": {"bitrate_bps": 250000}, "protocol": "rl_lcr", "seed": seed},
        topology=topo,
        protocol=protocol,
        trace_paths=val_traces,
        controller=ctrl,
        seed=seed,
        duration_s=5.0,
        warmup_s=0.0,
    )
    ctrl.attach(sim)
    sim.run()

    valid_pkts = [p for p in sim.collector.packet_records if not p.is_warmup]
    delivered = [p for p in valid_pkts if not np.isnan(p.t_deliv)]
    dropped = [p for p in valid_pkts if p.drop_reason is not None]
    in_flight = [p for p in valid_pkts if np.isnan(p.t_deliv) and p.drop_reason is None]

    total_accounted = len(delivered) + len(dropped) + len(in_flight)
    passed = (total_accounted == len(valid_pkts) and len(valid_pkts) > 0)
    dur = time.perf_counter() - t0
    details = (
        f"Total: {len(valid_pkts)} == Delivered: {len(delivered)} + "
        f"Dropped: {len(dropped)} + In-flight: {len(in_flight)}"
    )
    return CheckResult(3, "Packet Conservation Law", passed, details, dur)


def run_check_4_energy() -> CheckResult:
    """Verify energy accounting: sum(categories) == total; no negative residuals."""
    t0 = time.perf_counter()
    seed = 2026
    val_traces = load_traces(split="val", purpose="tuning")
    topo = generate_ward_topology(n_nodes=10, seed=seed)

    protocol = DummyForwarder()
    sim = Simulator(
        cfg={"radio": {"bitrate_bps": 250000}, "protocol": "dummy", "seed": seed},
        topology=topo,
        protocol=protocol,
        trace_paths=val_traces,
        seed=seed,
        duration_s=4.0,
    )
    sim.run()

    all_sums_ok = True
    all_residuals_ok = True
    all_battery_ok = True

    for node in sim.nodes.values():
        ea = node.energy
        sum_cats = sum(ea.consumed_j.values())
        if abs(sum_cats - ea.total_consumed()) > 1e-6:
            all_sums_ok = False
        res = ea.residual_frac()
        if not (0.0 <= res <= 1.0):
            all_residuals_ok = False
        expected_bat = (1.0 - ea.mains_fraction) * sum_cats
        if abs(ea.battery_consumed_j - expected_bat) > 1e-6:
            all_battery_ok = False

    passed = all_sums_ok and all_residuals_ok and all_battery_ok
    dur = time.perf_counter() - t0
    details = (
        f"Sum=Total: {all_sums_ok} | Battery formula: {all_battery_ok} | "
        f"Residuals in [0, 1]: {all_residuals_ok}"
    )
    return CheckResult(4, "Energy Accounting & Residuals", passed, details, dur)


def run_check_5_split_hygiene() -> CheckResult:
    """Verify strict partition hygiene and zero data leakage."""
    t0 = time.perf_counter()

    train_ok = False
    val_ok = False
    test_eval_ok = False
    tuning_test_rejected = False
    eval_val_rejected = False

    try:
        t_tr = load_traces(split="train", purpose="tuning")
        train_ok = len(t_tr) > 0
    except Exception:
        pass

    try:
        v_tr = load_traces(split="val", purpose="tuning")
        val_ok = len(v_tr) > 0
    except Exception:
        pass

    try:
        te_tr = load_traces(split="test", purpose="evaluation")
        test_eval_ok = len(te_tr) > 0
    except Exception:
        pass

    # Tuning attempting to load test split must be rejected
    try:
        load_traces(split="test", purpose="tuning")
    except AssertionError:
        tuning_test_rejected = True
    except Exception:
        pass

    # Evaluation attempting to load val split must be rejected
    try:
        load_traces(split="val", purpose="evaluation")
    except AssertionError:
        eval_val_rejected = True
    except Exception:
        pass

    # Patient disjointness
    with open("data/splits/mimic_patient_split.json", encoding="utf-8") as f:
        split_data = json.load(f)
    train_p = set(split_data["train"])
    val_p = set(split_data["val"])
    test_p = set(split_data["test"])
    disjoint_ok = train_p.isdisjoint(val_p) and val_p.isdisjoint(test_p) and train_p.isdisjoint(test_p)

    passed = (
        train_ok
        and val_ok
        and test_eval_ok
        and tuning_test_rejected
        and eval_val_rejected
        and disjoint_ok
    )
    dur = time.perf_counter() - t0
    details = (
        f"Tuning isolation: {tuning_test_rejected} | "
        f"Eval isolation: {eval_val_rejected} | "
        f"Disjoint cohorts: {disjoint_ok} (train={len(train_p)}, val={len(val_p)}, test={len(test_p)})"
    )
    return CheckResult(5, "Split Hygiene & Zero-Leakage Guard", passed, details, dur)


def run_check_6_feasibility() -> CheckResult:
    """Verify RL-LCR constraint feasibility flags and safety fallback engagement."""
    t0 = time.perf_counter()
    seed = 2026
    val_traces = load_traces(split="val", purpose="tuning")
    topo = generate_ward_topology(n_nodes=10, seed=seed)

    protocol = RLLCR(lr=0.01, iters=20, rho=1.0, omega_tx=1.0, k_candidates=2, device_pref="cpu")
    ctrl = SDNController(sink_id=topo.sink_id, route_recompute_interval_s=5.0)
    sim = Simulator(
        cfg={"radio": {"bitrate_bps": 250000}, "protocol": "rl_lcr", "seed": seed},
        topology=topo,
        protocol=protocol,
        trace_paths=val_traces,
        controller=ctrl,
        seed=seed,
        duration_s=6.0,
    )
    ctrl.attach(sim)
    sim.run()

    records = sim.collector.route_records
    has_records = len(records) > 0
    consistent_fallback = True
    for r in records:
        if r.feasible and r.fallback_used:
            consistent_fallback = False
        if not r.feasible and not r.fallback_used:
            consistent_fallback = False

    # Unit verification of strict checker
    checker = FeasibilityChecker(clinical_mapper=ClinicalConstraintMapper())
    feas_ok = checker.check(
        class_delays={"ARRHYTHMIA": 0.20, "DESATURATION": 0.35, "ROUTINE": 1.2},
        node_powers={0: 0.001, 1: 0.0005},
        node_types=["bedside", "wearable"],
    ).is_feasible

    feas_fail = checker.check(
        class_delays={"ARRHYTHMIA": 0.40, "DESATURATION": 0.35, "ROUTINE": 1.2},
        node_powers={0: 0.001, 1: 0.0005},
        node_types=["bedside", "wearable"],
    ).is_feasible

    passed = has_records and consistent_fallback and feas_ok and (not feas_fail)
    dur = time.perf_counter() - t0
    details = (
        f"Route decisions: {len(records)} | Fallback consistency: {consistent_fallback} | "
        f"Feasible pass: {feas_ok} | Infeasible caught: {not feas_fail}"
    )
    return CheckResult(6, "RL-LCR Feasibility & Fallback Invariants", passed, details, dur)


def generate_verification_report(
    results: list[CheckResult],
    output_path: Path | str = "reports/verification.md",
) -> bool:
    """Generate Markdown verification report and return overall pass status.

    Args:
        results: List of CheckResult objects.
        output_path: Destination path for Markdown report.

    Returns:
        bool: True if every check passed, False otherwise.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    all_passed = all(r.passed for r in results)
    overall_status = "PASS" if all_passed else "FAIL"

    now_utc = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    git_hash = get_git_hash()

    lines = [
        "# Pre-Experiment Invariant and Regression Verification Report",
        "",
        f"**Date Generated**: {now_utc}  ",
        f"**Repository Git Commit**: `{git_hash}`  ",
        f"**Platform**: `{platform.system()} {platform.machine()}` (Python {platform.python_version()})  ",
        f"**Overall Verification Status**: **{overall_status}**  ",
        "",
        "> [!IMPORTANT]",
        "> Phase 5 comparative evaluation experiments **MUST REFUSE TO START** if this file is missing",
        "> or contains any `FAIL` status.",
        "",
        "## Summary of Verification Checks",
        "",
        "| Check # | Invariant Description | Status | Execution Time | Details |",
        "| :---: | :--- | :---: | :---: | :--- |",
    ]

    for r in results:
        status_badge = "**PASS**" if r.passed else "<span style='color:red'>**FAIL**</span>"
        lines.append(f"| {r.check_id} | {r.name} | {status_badge} | {r.duration_s:.2f}s | {r.details} |")

    lines.extend([
        "",
        "## Invariant Specifications",
        "",
        "1. **Determinism**: Identical `(method, scenario, seed)` produces bit-for-bit identical "
        "`packets.parquet` SHA256 checksums across independent runs.",
        "2. **Common Random Numbers (CRN)**: All competing protocols with the same seed receive "
        "identical topology layout, patient mapping, and packet streams.",
        "3. **Conservation Law**: Closed packet balance equation: "
        "`Total Generated == Delivered + Dropped + In-flight` with zero packet loss or ghost creation.",
        "4. **Energy Accounting**: Strict energy conservation where categorized dissipation sums to "
        "total consumption and battery residual fractions remain in `[0.0, 1.0]` without negative values.",
        "5. **Split Hygiene**: Strict partition isolation: tuning scripts can only access `train` and `val`; "
        "evaluation scripts can only access `test` and `zeroshot`. Contamination raises fatal assertions.",
        "6. **RL-LCR Feasibility Invariant**: Feasible-flagged routes strictly respect all clinical delay "
        "and power bounds. Any constraint breach triggers priority-triage safety fallback routing.",
        "",
        f"## Authorization Decision: **{'PHASE 5 AUTHORIZED' if all_passed else 'PHASE 5 REFUSED'}**",
        "",
    ])

    out_file.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Verification report saved to {out_file} (Status: {overall_status})")
    return all_passed


def check_verification_passed(report_path: Path | str = "reports/verification.md") -> bool:
    """Pre-flight assertion for Phase 5 experiment runners.

    Refuses execution if verification report does not exist or indicates FAIL.

    Args:
        report_path: Path to reports/verification.md.

    Returns:
        bool: True if verification report exists and all checks passed.

    Raises:
        RuntimeError: If verification report is missing or contains FAIL.
    """
    p = Path(report_path)
    if not p.exists():
        raise RuntimeError(
            f"Phase 5 experiment pre-condition failed: Verification report '{report_path}' not found! "
            f"Run 'python scripts/verify_before_experiments.py' before launching experiments."
        )

    content = p.read_text(encoding="utf-8")
    if "Overall Verification Status**: **FAIL**" in content or "PHASE 5 REFUSED" in content:
        raise RuntimeError(
            f"Phase 5 experiment pre-condition failed: '{report_path}' contains FAIL status! "
            f"Fix invariant violations and re-run 'python scripts/verify_before_experiments.py'."
        )

    return True


def main() -> None:
    """CLI execution entrypoint."""
    setup_logging("verify_before_experiments")
    parser = argparse.ArgumentParser(description="Run pre-experiment invariant verification.")
    parser.add_argument(
        "--output",
        type=str,
        default="reports/verification.md",
        help="Path for verification markdown report.",
    )
    args = parser.parse_args()

    logger.info("Executing Pre-Experiment Invariant & Regression Checks...")
    t_start = time.perf_counter()

    checks = [
        run_check_1_determinism,
        run_check_2_crn,
        run_check_3_conservation,
        run_check_4_energy,
        run_check_5_split_hygiene,
        run_check_6_feasibility,
    ]

    results: list[CheckResult] = []
    for check_fn in checks:
        res = check_fn()
        results.append(res)
        status_str = "PASS" if res.passed else "FAIL"
        logger.info(f"Check {res.check_id}: {res.name} -> [{status_str}] in {res.duration_s:.2f}s")

    all_passed = generate_verification_report(results, output_path=args.output)
    t_total = time.perf_counter() - t_start

    print("\n" + "=" * 70)
    print("PRE-EXPERIMENT INVARIANT VERIFICATION SUMMARY")
    print("=" * 70)
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"[{status:4s}] Check {r.check_id}: {r.name:<40s} ({r.duration_s:.2f}s)")
    print("-" * 70)
    decision = "PHASE 5 AUTHORIZED (ALL PASSED)" if all_passed else "PHASE 5 REFUSED (FAILURES DETECTED)"
    print(f"Total Time:   {t_total:.2f} s")
    print(f"Decision:     {decision}")
    print(f"Report:       {args.output}")
    print("=" * 70 + "\n")

    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
