# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 18:48:34 2026

@author: royi.b
Association module for target tracking and range unfolding.
"""
import numpy as np
from collections import defaultdict


def group_plots_by_pri(plot_manager):
    """
    Groups detections by PRI_us.
    """
    grouped = defaultdict(list)

    for plot in plot_manager:
        grouped[plot["PRI_us"]].append(plot)

    return grouped

def generate_unfold_hypotheses_from_plots(plot_manager, fc, max_range_zones=12, vel_zones=(-1, 0, 1)):
    """
    Generates unfolded (R, V) hypotheses from plot_manager.
    """
    c = 3e8;
    lam = c / fc
    plots_by_pri = group_plots_by_pri(plot_manager)

    hypotheses = []

    for PRI_us, plots in plots_by_pri.items():
        PRI = PRI_us * 1e-6
        Ru = c * PRI / 2
        Vu = lam / (2 * PRI)

        for plot in plots:
            amb_rng = plot["amb_rng"]
            amb_vel = plot["amb_vel"]

            for k in range(max_range_zones):
                for m in vel_zones:
                    hypotheses.append({
                        "PRI_us": PRI_us,
                        "R": amb_rng + k * Ru,
                        "V": amb_vel + m * Vu,
                        "CUT_dB": plot["CUT_dB"],
                        "plot": plot   # back-reference
                    })

    return hypotheses

def associate_unfolded_targets(hypotheses, r_tol=50.0, v_tol=2.0, min_pri_support=3):
    """
    Associates unfolded hypotheses into targets.
    """
    clusters = []

    for h in hypotheses:
        assigned = False

        for c in clusters:
            if h["PRI_us"] in c["PRI_us_set"]:
                continue

            if (abs(h["R"] - c["R_mean"]) < r_tol and
                abs(h["V"] - c["V_mean"]) < v_tol):

                c["members"].append(h)
                c["PRI_us_set"].add(h["PRI_us"])

                weights = [10**(m["CUT_dB"]/10) for m in c["members"]]
                c["R_mean"] = np.average([m["R"] for m in c["members"]], weights=weights)
                c["V_mean"] = np.average([m["V"] for m in c["members"]], weights=weights)

                assigned = True
                break

        if not assigned:
            clusters.append({
                "members": [h],
                "PRI_us_set": {h["PRI_us"]},
                "R_mean": h["R"],
                "V_mean": h["V"]
            })

    # Filter valid unfolded targets
    targets = []
    for c in clusters:
        if len(c["PRI_us_set"]) >= min_pri_support:
            targets.append({
                "filt_rng": c["R_mean"],
                "filt_vel": c["V_mean"],
                "num_pri": len(c["PRI_us_set"]),
                "mean_power_dB": np.mean([m["CUT_dB"] for m in c["members"]]),
                "supporting_plots": [m["plot"] for m in c["members"]]
            })

    return targets

def unfold_targets_from_plot_manager(plot_manager, fc, r_tol=50.0, v_tol=2.0, min_pri_support=3):
    """
    Full unfolding pipeline from plot_manager.
    """
    hypotheses = generate_unfold_hypotheses_from_plots(plot_manager, fc)

    targets = associate_unfolded_targets(
        hypotheses,
        r_tol=r_tol,
        v_tol=v_tol,
        min_pri_support=min_pri_support)

    return targets

def update_target_manager(target_manager, unfolded_targets, cycle_idx, r_gate=150.0, v_gate=5.0, max_missed=3):
    """
    Updates persistent targets with newly unfolded targets.
    """
    assigned = set()

    # --- Update existing targets ---
    for tgt in target_manager:
        best = None
        best_dist = np.inf

        for i, meas in enumerate(unfolded_targets):
            if i in assigned:
                continue

            dr = abs(meas["filt_rng"] - tgt["filt_rng"])
            dv = abs(meas["filt_vel"] - tgt["filt_vel"])

            if dr < r_gate and dv < v_gate:
                dist = dr + 10 * dv
                if dist < best_dist:
                    best = i
                    best_dist = dist

        if best is not None:
            meas = unfolded_targets[best]

            # simple exponential smoothing
            alpha = 0.4
            tgt["filt_rng"] = (1 - alpha) * tgt["filt_rng"] + alpha * meas["filt_rng"]
            tgt["filt_vel"] = (1 - alpha) * tgt["filt_vel"] + alpha * meas["filt_vel"]
            tgt["power_dB"] = meas["mean_power_dB"]

            tgt["last_update"] = cycle_idx
            tgt["hits"] += 1
            tgt["age"] += 1

            assigned.add(best)
        else:
            tgt["age"] += 1

    # --- Initiate new targets ---
    next_id = max([t["id"] for t in target_manager], default=0) + 1

    for i, meas in enumerate(unfolded_targets):
        if i in assigned:
            continue

        target_manager.append({
            "id": next_id,
            "filt_rng": meas["filt_rng"],
            "filt_vel": meas["filt_vel"],
            "power_dB": meas["mean_power_dB"],
            "last_update": cycle_idx,
            "age": 1,
            "hits": 1
        })
        next_id += 1

    # --- Prune stale targets ---
    target_manager[:] = [
        t for t in target_manager
        if cycle_idx - t["last_update"] <= max_missed
    ]