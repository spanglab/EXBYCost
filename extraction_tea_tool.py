import streamlit as st
import os

import platform
if platform.system() == "Windows":
    os.environ["PATH"] += os.pathsep + r"C:\Program Files\Graphviz\bin"
import biosteam as bst
import thermosteam as tmo
from thermosteam import Chemical, MultiStream, separations
from biosteam import units
from biosteam.units.decorators import cost
from math import exp, log, log10
import math as _math
import flexsolve as flx
import numpy as np
from scipy.optimize import brentq
from SolidSolventExtractor import SolidSolventExtractor
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from Mill import Mill
from biosteam_proxy_finder import assign_proxies
from pricing import (get_solvent_prices, get_california_industrial_price,
                     get_california_operator_wage)
from temperature_thresholds import (
    ChebiResolver, ThresholdClassifier, evaluate_temperature_flags,
    flags_to_rows, narrative_lines, build_chem_meta,
    normalize_name as _tt_norm)
import io
import time
import itertools
import contextlib
import threading
import json
import tempfile
import hashlib
import warnings
warnings.filterwarnings("ignore", message=".*has no defined Dortmund groups.*")
warnings.filterwarnings("ignore", message=".*overflow encountered in exp.*")

# ============================================================================
# Evaporator + Spray Dryer classes
# ============================================================================

def suggest_pressures(solvent_ID, n_effects=3, T_high_C=None, delta_T_C=20,
                      feed_T_C=None, margin_C=2.0):
    """Generate evaporator pressure tuple from solvent Psat curve.

    The first effect is normally placed at the solvent atmospheric boiling
    point (Tb). If the feed arrives hotter than Tb (feed_T_C above Tb), the
    first effect is instead placed at the feed temperature plus a small
    margin, so the evaporator runs pressurised at the feed boiling point
    rather than cooling the feed. This keeps the feed saturated rather than
    superheated and avoids the condenser inverted-temperature spec. Feeds at
    or below Tb are unaffected (the max keeps T_high_C at Tb).
    """
    chem = tmo.settings.chemicals[solvent_ID]
    Tb_C = chem.Tb - 273.15
    if T_high_C is None:
        T_high_C = Tb_C
    if feed_T_C is not None:
        # First effect must boil at or above the feed temperature.
        T_high_C = max(T_high_C, feed_T_C + margin_C)
    temps_K = [T_high_C + 273.15 - i * delta_T_C for i in range(n_effects)]
    pressures = [chem.Psat(T) for T in temps_K]
    return tuple(round(p) for p in pressures)


class CorrectedMEE(bst.MultiEffectEvaporator):
    """
    MultiEffectEvaporator with corrected flash=True output assignment.

    Fixes BioSTEAM's V_overall misinterpretation in the final output
    streams while keeping the multi-stage calculation intact for
    correct cost and energy results.
    """

    # ---- Diagnostics -------------------------------------------------------
    # _flash_fallback_used is reset on EVERY _run, and _run is called dozens to
    # thousands of times per sys.simulate() (once per V-bisection step, per
    # Wegstein pass). Reading it after simulate() therefore only reports the
    # LAST call and hides everything that happened on the way. These counters
    # accumulate instead, and are never reset by _run -- call reset_counters()
    # once, just after the unit is built.
    def reset_counters(self):
        self._run_count = 0
        self._guard_counts = {}
        self._guard_reasons = {}

    def _note_guard(self, name, exc=None):
        if not hasattr(self, '_guard_counts'):
            self.reset_counters()
        self._guard_counts[name] = self._guard_counts.get(name, 0) + 1
        if exc is not None:
            msg = f"{type(exc).__name__}: {exc}"
            self._guard_reasons.setdefault(name, [])
            if msg not in self._guard_reasons[name]:
                self._guard_reasons[name].append(msg)

    @property
    def n_run(self):
        """Total _run() calls since the last reset_counters()."""
        return getattr(self, '_run_count', 0)

    @property
    def n_fb(self):
        """Total guard trips since the last reset_counters() (all guards)."""
        return sum(getattr(self, '_guard_counts', {}).values())

    # ---- Design ------------------------------------------------------------
    def _design(self):
        """Size the evaporator.

        BioSTEAM's MultiEffectEvaporator._design reads EVERYTHING from the
        cascade objects:
            duty_1  <- first_evaporator.H_out - first_evaporator.H_in
            duty_i  <- evap.design_results['Heat transfer']
            LMTD_i  <- evap.outs[0].T, evap.outs[2].T
            volume  <- evap._size_flash_vessel()
        When a guard fires, those objects hold whatever half-solved state the
        crashed cascade left behind -- and _design reads it as though it were a
        solution. That is how a guard trip produced a complete, plausible, and
        meaningless capital cost.

        So: if the cascade solved AND the resulting design is physically
        sane, use BioSTEAM's rigorous design, unchanged. If a guard fired,
        OR the rigorous design produced something non-finite / negative
        (which can happen with no exception at all -- see below), discard
        the cascade state and size from an explicit energy balance instead.

        The second condition exists because a guard trip is not the only way
        the cascade can go wrong. At extreme V (near-total vaporization,
        which is what a very high recycle cap pushes the evaporator toward),
        IQ_interpolation can converge -- no exception, _flash_fallback_used
        stays False -- while still landing on a per-effect duty split that is
        physically degenerate: one effect gets assigned a NEGATIVE area.
        BioSTEAM's own cost correlation does log(A * 0.0929), which is
        undefined for negative A, so this produces a silent NaN that nothing
        upstream flagged as a failure:
            CostWarning: heat transfer area requirement (-0.207 ft2) ...
            RuntimeWarning: invalid value encountered in log
        So the rigorous path's OUTPUT is validated, not just whether it threw.
        """
        if not getattr(self, '_flash_fallback_used', False):
            self._design_from_shortcut = False
            super()._design()

            # A single effect can come back with a negative / non-finite area
            # (e.g. -0.207 ft2 from a slight temperature inversion in a
            # lightly-loaded effect near total vaporization). BioSTEAM's cost
            # correlation does log(A), so that one effect poisons the whole
            # evaporator cost with NaN. The other effects are usually fine.
            #
            # Rather than discard the entire rigorous cascade, zero out ONLY
            # the degenerate effect(s) — an effect with an inverted/near-zero
            # driving force carries no useful duty, so contributing 0 area and
            # 0 cost is the right accounting — and keep every healthy effect's
            # rigorous sizing. This is what BioSTEAM already does for effects
            # with Q <= 1e-12; it just misses the negative-LMTD case. No
            # re-simulation: this only rewrites the per-effect area/cost lists.
            _As = getattr(self, '_As', None)
            _costs = getattr(self, '_evap_costs', None)
            _zeroed = []
            if _As is not None and _costs is not None:
                for _i in range(len(_As)):
                    _a = _As[_i]
                    _c = _costs[_i] if _i < len(_costs) else 0.0
                    if (not _math.isfinite(_a) or _a < 0
                            or not _math.isfinite(_c) or _c < 0):
                        _zeroed.append(_i)
                        _As[_i] = 0.0
                        if _i < len(_costs):
                            _costs[_i] = 0.0
                if _zeroed:
                    self._As = _As
                    self._evap_costs = _costs
                    self.design_results['Area'] = sum(_As)
                    self.baseline_purchase_costs['Evaporators'] = sum(_costs)
                    self._degenerate_effects = _zeroed
                    self._note_guard(
                        'degenerate effect zeroed (negative/non-finite area)',
                        ValueError(f"effects {_zeroed} of {len(_As)} zeroed; "
                                   f"As now {[round(a, 3) for a in _As]}"))

            # Safety net: if EVERY effect was degenerate (nothing left to
            # cost), or the totals are still non-finite for any other reason,
            # fall back to the bounded equal-split energy-balance design.
            _area = self.design_results.get('Area', None)
            _cost = self.baseline_purchase_costs.get('Evaporators', None)
            _all_zero = _As is not None and len(_zeroed) == len(_As)
            if (_all_zero
                    or (_area is not None
                        and (not _math.isfinite(_area) or _area < 0))
                    or (_cost is not None
                        and (not _math.isfinite(_cost) or _cost < 0))):
                self._note_guard(
                    'rigorous design unusable -> energy-balance shortcut',
                    ValueError(f"Area={_area}, Cost={_cost}"))
                self._flash_fallback_used = True
                self._design_from_shortcut = True
                self._design_from_energy_balance()
            return
        self._design_from_shortcut = True
        self._design_from_energy_balance()

    def _design_from_energy_balance(self):
        """Short-cut N-effect design, used ONLY when the cascade failed.

        Reads only the inlet/outlet streams and the SPECIFIED pressures --
        never the (crashed) cascade objects. This is the standard hand
        calculation for a multi-effect evaporator, not a fudge:

          * The outlet streams are trustworthy even on the fallback path: the
            flash=True block reassigns them from V_overall, and
            _assign_limiting_split assigns them by pure mass balance. So the
            solvent evaporated is known exactly.
          * Effect temperatures come from the specified pressures via Tsat.
          * Steam economy: in an N-effect, each effect's vapour is the next
            effect's heating medium, so each effect evaporates ~1/N of the
            total and only effect 1 draws external utility.
          * A_i = Q_i / (U * dT_i), with dT_1 = T_steam - T_1 and
            dT_i = T_(i-1) - T_i. Both sides isothermal, so LMTD = dT.
          * Cost uses the SAME C_func / U / area correlation as the rigorous
            path, so the two are directly comparable.

        Validated against the rigorous design on cases that solve cleanly:
        area within ~4-8%, cost within ~2-4% at V = 0.9, erring HIGH (so it
        does not understate CAPEX). Accuracy degrades at low V where the
        equal-split assumption is poorest -- but the fallback only fires as
        V -> 1, which is where the approximation is at its best.
        """
        from biosteam.units.design_tools import heat_transfer as _ht

        A_range, C_func, U, _mat = self._evap_data
        Design = self.design_results
        Cost = self.baseline_purchase_costs
        CE = bst.CE

        chem_id = self.chemical
        chem = self.chemicals[chem_id]
        feed = self.ins[0]
        conc = self.outs[0]

        # 1. Solvent actually evaporated -- from the OUTLET streams (correct),
        #    not from the cascade (crashed).
        n_evap = max(0.0, feed.imol[chem_id] - conc.imol[chem_id])   # kmol/hr
        N = max(1, len(self.evaporators))

        if n_evap <= 1e-12:
            Design['Area'] = 0.0
            Design['Volume'] = 0.0
            Cost['Evaporators'] = 0.0
            self._As = [0.0] * N
            return

        # 2. Effect temperatures from the SPECIFIED pressures.
        P_list = list(self.P)[:N]
        T_eff = []
        for P in P_list:
            try:
                T_eff.append(chem.Tsat(P))
            except Exception:
                T_eff.append(chem.Tb)
        N = len(T_eff)

        def _hvap(T):
            Tc = getattr(chem, 'Tc', None)
            if Tc:
                T = min(T, Tc - 1.0)
            return chem.Hvap(T)          # J/mol

        # 3. Duties. kmol/hr * J/mol == kJ/hr, so no conversion needed.
        n_per = n_evap / N
        Q_lat = [n_per * _hvap(T) for T in T_eff]
        Q_sens = feed.C * max(0.0, T_eff[0] - feed.T)      # feed.C: kJ/hr/K
        Q1 = Q_lat[0] + Q_sens

        # 4. External utility supplies effect 1 only.
        hu = self.create_heat_utility()
        hu(Q1, feed.T, T_eff[0])
        try:
            Th = hu.inlet_utility_stream.T
        except Exception:
            Th = T_eff[0] + 20.0

        # 5. Areas and costs -- same correlation as the rigorous path.
        #    Validated at each step: this is the path that is SUPPOSED to be
        #    the safety net, so it must not itself be able to emit a NaN. If
        #    a duty or LMTD comes out non-finite (e.g. Hvap undefined this
        #    close to Tc for some solvent/pressure combination), fail loudly
        #    here rather than silently handing a NaN to the TEA.
        As, evap_costs = [], []
        bounds_warning = bst.exceptions.bounds_warning
        for i, T in enumerate(T_eff):
            Qi = Q1 if i == 0 else Q_lat[i]
            T_hot = Th if i == 0 else T_eff[i - 1]
            LMTD = max(T_hot - T, 1.0)    # isothermal both sides -> LMTD = dT
            if not (_math.isfinite(Qi) and Qi >= 0 and _math.isfinite(LMTD)):
                raise RuntimeError(
                    f"CorrectedMEE energy-balance sizing produced a "
                    f"non-finite/negative duty on effect {i + 1} "
                    f"(Q={Qi}, LMTD={LMTD}, T_eff={T_eff}, P={P_list}). "
                    f"This is the sizing fallback itself failing; the "
                    f"pressures or solvent/temperature combination need "
                    f"checking directly.")
            A = abs(_ht.compute_heat_transfer_area(LMTD, U, Qi, 1.))
            if not _math.isfinite(A):
                raise RuntimeError(
                    f"CorrectedMEE energy-balance sizing produced a "
                    f"non-finite area on effect {i + 1} (Q={Qi}, LMTD={LMTD}).")
            As.append(A)
            bounds_warning(self, 'heat transfer area requirement', A, 'ft2',
                           A_range, 'cost')
            _c = C_func(A, CE)
            if not _math.isfinite(_c):
                raise RuntimeError(
                    f"CorrectedMEE energy-balance sizing: cost correlation "
                    f"returned non-finite for effect {i + 1} (A={A:.4g} ft2, "
                    f"outside its valid range {A_range}). Area is far enough "
                    f"outside the correlation's fitted range that the "
                    f"polynomial itself blows up; check flow scale / N "
                    f"effects for this case.")
            evap_costs.append(_c)

        self._As = As
        self._evap_costs = evap_costs
        Design['Area'] = sum(As)
        Cost['Evaporators'] = sum(evap_costs)

        # 6. Vessel volume for the vacuum system. _size_flash_vessel() would
        #    read the crashed cascade, so estimate from the vapour volumetric
        #    rate at ~30 s disengagement residence. Rough, but it only feeds
        #    the vacuum-system cost, a second-order term.
        R, tau = 8.314, 30.0 / 3600.0     # J/mol/K, hr
        total_volume = 0.0
        for T, P in zip(T_eff, P_list):
            if P > 0:
                total_volume += n_per * 1000.0 * R * T / P * tau   # m3
        Design['Volume'] = total_volume

        # 7. Condenser / vacuum system: attempt normal sizing; if they also
        #    fail, record it rather than crashing the whole design step.
        try:
            self.condenser.simulate(run=False)
        except Exception as e:
            self._note_guard('condenser design (shortcut path)', e)
        try:
            self.vacuum_system = bst.VacuumSystem(
                self, self.vacuum_system_preference,
                vessel_volume=total_volume, P_suction=self.outs[0].P)
        except Exception as e:
            self._note_guard('vacuum system (shortcut path)', e)

    def _run(self):
        out_wt_solids, liq = self.outs
        ins = self.ins
        self._run_count = getattr(self, '_run_count', 0) + 1
        self._flash_fallback_used = False
        self._vub_direct_bound_used = False
        if self.V == 0:
            out_wt_solids.copy_like(ins[0])
            liq.empty()
            self._reload_components = True
            return
        if self._reload_components:
            self._load_components()
            self._reload_components = False

        if self.V_definition == 'Overall':
            P = tuple(self.P)
            try:
                self.P = list(P)
                for i in range(self._N_evap - 1):
                    if self._V_overall(0.) > self.V:
                        self.P.pop()
                        self._load_components()
                        self._reload_components = True
                    else:
                        break
                self.P = P
                n_eff = max(1, self._N_evap)
                f = self._V_overall_objective_function
                V_ceiling = 0.999
                # Bound GUARANTEED to bracket from above: overall vaporization
                # is always >= the first-effect fraction, so f(V_first=self.V)
                # is >= 0.
                V_target = min(self.V, V_ceiling)

                # SEED. f(V_first) is monotonically increasing and the pop
                # loop above guarantees f(0) <= 0, so all that is needed is a
                # good upper probe.
                #
                # The old heuristic (min(V, V/n_eff*1.5, 0.35)) assumed each
                # effect vaporizes an equal SLICE of the ORIGINAL feed. The
                # cascade is multiplicative, not additive, so that assumption
                # collapses as V -> 1: at V=0.99, n_eff=3 it probes 0.35 while
                # the root sits near 0.79 -- the probe is wasted and
                # IQ_interpolation then gets handed the whole [0, 0.99]
                # interval with a useless midpoint guess.
                #
                # Equal-FRACTION staging instead gives
                #     (1 - V_overall) = (1 - V_first) ** n_eff
                # which stays valid as V -> 1 and is slightly TIGHTER than the
                # old cap for concentrated feeds, so nothing regresses there.
                # No safety margin is applied -- biasing the seed LOW is
                # deliberate, because a low probe is recycled as a lower bound
                # (below) whereas a high probe just leaves a wide bracket.
                V_seed = 1.0 - (1.0 - V_target) ** (1.0 / n_eff)
                V_seed = min(V_target, max(V_seed, 1e-4))

                V_lo = 0.0
                f_at_zero = f(0.0)
                f_lo = f_at_zero
                V_ub = V_seed
                f_hi = f(V_seed)

                if f_lo * f_hi > 0.0:
                    # Seed landed below the root. Since f is increasing, that
                    # failed probe is NOT wasted -- it is a valid new LOWER
                    # bound. Keep it (5x narrower bracket for IQ) and raise the
                    # ceiling to the guaranteed bound.
                    if f_hi > f_lo:                  # monotonicity holds
                        V_lo, f_lo = V_ub, f_hi
                    V_ub = V_target
                    f_hi = f(V_ub)
                    self._vub_direct_bound_used = True

                # Safety net: if the guaranteed bound somehow does not bracket
                # (numerical edge), fall back to the original outward search.
                # First reopen the tightened lower bound -- it is only sound if
                # the cascade really is monotonic, and f(0) is already cached
                # so reopening costs nothing.
                if f_lo * f_hi > 0.0 and V_lo > 0.0:
                    V_lo, f_lo = 0.0, f_at_zero
                while f_lo * f_hi > 0.0 and V_ub < V_ceiling:
                    V_ub = min(V_ceiling, (V_ub * 1.5) if V_ub > 1e-6 else 0.05)
                    try:
                        f_hi = f(V_ub)
                    except Exception as e:
                        self._note_guard('bracket-search abort', e)
                        break
                if f_lo * f_hi > 0.0:
                    # Target overall V unreachable even at the ceiling: clamp
                    # to the closest feasible bound (records a result rather
                    # than crashing; concentrate misses its solids target).
                    self._V_first_effect = V_ub if abs(f_hi) < abs(f_lo) else V_lo
                    f(self._V_first_effect)  # restore cascade state at pick
                else:
                    guess = self._V_first_effect
                    if guess is None or not (V_lo <= guess <= V_ub):
                        guess = 0.5 * (V_lo + V_ub)
                    self._V_first_effect = flx.IQ_interpolation(
                        f, V_lo, V_ub, f_lo, f_hi, guess,
                        xtol=1e-4, ytol=1e-3, checkiter=False)
            except Exception as e:
                # GUARD 1 (V-solve). Reached ONLY when the proper solve above
                # raised -- typically a VLE flash inside _V_overall that could
                # not bracket against the V=1 edge ("failed to find bracket")
                # for a volatile solvent carrying non-volatile solute. Points
                # that solve normally never enter here. Keep the target
                # overall V (what the output split needs) and set the
                # first-effect fraction to a plausible estimate so cost/energy
                # stay finite. Uses the same equal-FRACTION staging estimate as
                # the seed above rather than the old additive cap, which
                # collapsed to a wildly low 0.35 for dilute (V -> 1) feeds --
                # i.e. exactly the feeds most likely to land in this guard.
                self.P = P
                try:
                    self._reload_components = True
                    self._load_components()
                except Exception as e2:
                    self._note_guard('GUARD 1 inner (_load_components)', e2)
                n_eff = max(1, self._N_evap)
                _V_t = min(self.V, 0.999)
                self._V_first_effect = min(
                    self.V, 1.0 - (1.0 - _V_t) ** (1.0 / n_eff))
                self._flash_fallback_used = True
                self._note_guard('GUARD 1 (V-solve)', e)
                warnings.warn(
                    "CorrectedMEE: V-solve flash did not converge "
                    f"({type(e).__name__}: {e}); using limiting-split "
                    "fallback for this point.")
            V_overall = self.V
        else:
            V_overall = self._V_overall(self.V)

        evaporators = self.evaporators
        last_evaporator = evaporators[-1]

        def _assign_limiting_split():
            # Unambiguous limiting split for a degenerate near-complete
            # vaporization: the volatile solvent boils off to the target
            # remaining amount, every non-volatile component stays in the
            # liquid concentrate. Pure mass split (no VLE call) so it cannot
            # itself raise. Used only by the fallback branches.
            chem_id = self.chemical
            feed_s = self.ins[0]
            solvent_in = feed_s.imol[chem_id]
            liq_solvent = min(solvent_in,
                              max(0.0, solvent_in * (1.0 - V_overall)))
            vap_solvent = max(0.0, solvent_in - liq_solvent)
            out_wt_solids.copy_like(feed_s)
            out_wt_solids.imol[chem_id] = liq_solvent
            out_wt_solids.P = self.ins[0].P
            liq.empty()
            liq.phase = 'g'
            liq.imol[chem_id] = vap_solvent
            liq.T = feed_s.T
            liq.P = self.ins[0].P

        try:
            self.condenser._run()
            liq = self.mixer.outs[0]
            liq.P = self.ins[0].P
            liq.mix_from(self.mixer.ins, conserve_phases=True)
        except Exception as e:
            # GUARD 2 (condenser/mixer). Only if condensing the combined
            # vapor failed to converge. Assign the limiting split and finish.
            self._flash_fallback_used = True
            self._note_guard('GUARD 2 (condenser/mixer)', e)
            warnings.warn(
                "CorrectedMEE: condenser flash did not converge "
                f"({type(e).__name__}: {e}); using limiting-split fallback.")
            _assign_limiting_split()
            liq.P = out_wt_solids.P
            return

        # --- CORRECTED VLE output assignment ---
        if self.flash:
            chemical = self.chemical
            feed = self.ins[0]
            target_remaining = feed.imol[chemical] * (1.0 - V_overall)
            mixed_stream = MultiStream(None, thermo=self.thermo)

            total_mol = feed.F_mol
            if total_mol > 0:
                V_max = feed.imol[chemical] / total_mol * 0.999
            else:
                V_max = 0.9

            def objective(V_total):
                mixed_stream.copy_flow(feed)
                mixed_stream.vle(P=last_evaporator.P, V=V_total)
                return mixed_stream.imol['l', chemical] - target_remaining

            try:
                flx.IQ_interpolation(
                    objective, 0., V_max,
                    xtol=1e-4, ytol=1e-3, checkiter=False)
            except Exception as e_iq:
                self._note_guard('output-flash bisection rescue', e_iq)
                try:
                    # Existing bisection rescue first: handles non-degenerate
                    # IQ_interpolation misses without changing their outcome.
                    lo, hi = 0.0, V_max
                    for _ in range(15):
                        mid = (lo + hi) / 2
                        if objective(mid) > 0:
                            lo = mid
                        else:
                            hi = mid
                    mixed_stream.copy_flow(feed)
                    mixed_stream.vle(P=last_evaporator.P, V=(lo + hi) / 2)
                except Exception as e:
                    # GUARD 3 (final flash). Both the solver and the bisection
                    # re-flash failed -> genuinely degenerate flash. Assign the
                    # limiting split directly and finish.
                    self._flash_fallback_used = True
                    self._note_guard('GUARD 3 (final flash)', e)
                    warnings.warn(
                        "CorrectedMEE: output flash did not converge "
                        f"({type(e).__name__}: {e}); using limiting-split "
                        "fallback.")
                    _assign_limiting_split()
                    liq.P = out_wt_solids.P
                    return

            out_wt_solids.mol = mixed_stream.imol['l']
            if liq.phase == 'l':
                liq.phase = 'l'
                liq.mol = mixed_stream.imol['g']
            else:
                H = liq.H
                liq.copy_like(mixed_stream['g'])
                liq.vle(H=H, P=self.ins[0].P)

        liq.P = out_wt_solids.P


@cost('Evaporation rate', CE=567, units='lb/hr', BM=2.06,
      f=lambda W: exp(8.5133 + 0.9847*(logW:=log(W)) - 0.0561*logW*logW))
class SolventSprayDryer(bst.Unit):
    """
    Spray dryer with configurable solvent, energy balance, and
    thermal efficiency.

    Default T = solvent Tb + 10 K (Maas et al. 2014).
    Default efficiency = 0.218 (Patel & Bade 2019).
    """
    _units = {'Evaporation rate': 'lb/hr'}
    _N_ins = 1
    _N_outs = 2

    def _init(self, solvent='Water', moisture_content=0.05, T=None,
              thermal_efficiency=0.218):
        self.solvent = solvent
        self.moisture_content = moisture_content
        if T is None:
            self.T = tmo.settings.chemicals[solvent].Tb + 10
        else:
            self.T = T
        self.thermal_efficiency = thermal_efficiency

    def _run(self):
        feed = self.ins[0]
        vapor, solids = self.outs
        # Dry at no less than the feed temperature. If the concentrate
        # arrives hotter than the dryer setpoint (Tb + 10 K) -- e.g. a hot
        # acetone feed from a pressurised evaporator -- operate at the feed
        # temperature rather than trying to cool to the setpoint, which
        # would hand the heat utility an inverted ('inlet must be cooler
        # than outlet if heating') spec. Cooler feeds keep the setpoint.
        T_out = max(self.T, feed.T)
        solids.copy_like(feed)
        vapor.empty()
        if solids.F_mass <= 0:
            vapor.phase = 'g'; return
        current_frac = solids.imass[self.solvent] / solids.F_mass
        if current_frac <= self.moisture_content:
            vapor.phase = 'g'; vapor.T = T_out; solids.T = T_out; return
        vapor.copy_flow(solids, self.solvent, remove=True)
        separations.adjust_moisture_content(
            solids, vapor, self.moisture_content, ID=self.solvent)
        vapor.phase = 'g'
        vapor.T = T_out; solids.T = T_out
        solids.P = feed.P; vapor.P = feed.P

    def _design(self):
        self.design_results['Evaporation rate'] = max(
            self.outs[0].get_total_flow('lb/hr'), 0.001)
        duty = self.H_out - self.H_in
        if abs(duty) > 0:
            actual_duty = duty / self.thermal_efficiency
            T_out = max(self.T, self.ins[0].T)
            self.add_heat_utility(actual_duty, T_in=self.ins[0].T, T_out=T_out)


# ============================================================================
# Solid Cooler (indirect rotary drum cooler)
# ============================================================================
@cost('Heat transfer area', CE=567, units='ft^2', BM=2.06,
      cost=1520, S=1, n=0.80)
class SolidCooler(bst.Unit):
    """
    Indirect rotary drum cooler for bulk solids / dried powders.

    Cools a solid (or semi-solid) stream to a target temperature using
    cooling water circulated through a jacket or internal tubes in a
    rotating drum.

    Parameters
    ----------
    T_target : float
        Target outlet temperature [K].  Default 298.15 K (25 °C).
    U : float
        Overall heat-transfer coefficient [W/(m²·K)].
        Default 25 — typical for indirect water-jacketed rotary
        coolers handling fine powders.  Literature range 13–50 W/(m²·K).

    Cost basis
    ----------
    Indirect-heat steam-tube rotary dryer/cooler from
    Seider, Seader, Lewin & Widagdo (2017), Table 16.32:

        Cp = 1,520 · A^0.80   [USD, stainless steel]

    where A = heat-transfer area [ft²].
    Valid range: 100–1,400 ft².  CE = 567, BM = 2.06.

    References
    ----------
    [1] Seider et al. (2017), Product & Process Design Principles, 4th ed.
    [2] Nhuchhen et al. (2016), Int. J. Heat Mass Transf. 102, 64–76.
    """
    _units = {'Heat transfer area': 'ft^2'}
    _N_ins = 1
    _N_outs = 1

    def _init(self, T_target=298.15, U=25):
        self.T_target = T_target
        self.U = U

    def _run(self):
        feed = self.ins[0]
        out = self.outs[0]
        out.copy_like(feed)
        if feed.T > self.T_target:
            out.T = self.T_target

    def _design(self):
        feed = self.ins[0]
        out = self.outs[0]
        duty = self.H_out - self.H_in
        Q_W = abs(duty) * 1000 / 3600

        if Q_W < 1.0:
            self.design_results['Heat transfer area'] = 0.001
            return

        T_hot_in  = feed.T
        T_hot_out = out.T
        T_cw_in   = 298.15
        T_cw_out  = min(T_hot_in - 5.0, 318.15)

        dT1 = T_hot_in  - T_cw_out
        dT2 = T_hot_out - T_cw_in

        if dT1 <= 0 or dT2 <= 0:
            LMTD = max(dT1, dT2, 1.0)
        elif abs(dT1 - dT2) < 0.01:
            LMTD = dT1
        else:
            LMTD = (dT1 - dT2) / log(dT1 / dT2)

        A_m2  = Q_W / (self.U * max(LMTD, 1.0))
        A_ft2 = A_m2 * 10.7639

        self.design_results['Heat transfer area'] = max(A_ft2, 0.001)
        self.add_heat_utility(duty, T_in=T_hot_in, T_out=T_hot_out)


# ============================================================================
# Feedstock Drum Dryer (indirect steam-tube rotary drum dryer)
# ============================================================================
@cost('Heat transfer area', CE=567, units='ft^2', BM=2.06,
      cost=1520, S=1, n=0.80)
class FeedDrumDryer(bst.Unit):
    """
    Rotary drum dryer for the incoming (as-received) feedstock.

    Placed upstream of feedstock storage, this unit dries the wet biomass
    down to a target moisture content before it goes into the storage tank.
    Two reasons this matters in practice:

    1. Storage stability. Wet biomass stored in bulk moulds, self-heats and
       loses extractable compounds. Below ~10 wt% moisture microbial and
       enzymatic activity is largely arrested; ~6 wt% is the usual target
       for stable long-term storage of dried botanical material
       [Chen et al. 2015; Emery & Mosier 2012].
    2. Water carried into the process. Feed moisture is a non-solvent
       impurity: it dilutes the extract, loads the evaporator, raises the
       purge required to hold the recycle-loop impurity cap, and ends up
       as residual water in the dried product. Removing it here, once,
       with a cheap bulk dryer is far cheaper than removing it repeatedly
       from the solvent loop.

    Moisture is specified on a wet basis (kg water / kg wet solids), the
    same convention as `SolventSprayDryer.moisture_content`.

    Parameters
    ----------
    moisture_ID : str
        Chemical evaporated. Default 'Water'.
    moisture_content : float
        Target outlet moisture, wet basis [kg/kg]. Default 0.06 (6 wt%).
    T : float
        Solids/vapor outlet temperature [K]. Default 363.15 K (90 degC).
    thermal_efficiency : float
        Fraction of the supplied heat that reaches the solids. Default 0.70,
        typical for indirect steam-tube rotary dryers (0.6-0.85)
        [Mujumdar, Handbook of Industrial Drying, 4th ed., Ch. 2].
    U : float
        Overall heat-transfer coefficient [W/(m2 K)] on the tube area.
        Default 60 - mid-range for steam-tube rotary dryers on wet granular
        solids (literature 30-85 W/(m2 K)) [Perry's Ch. 12].
    T_medium : float | None
        Heating-medium temperature [K] used for the LMTD when sizing the
        tube area. Default None -> taken from the selected heating agent,
        clamped to [T_out + 10 K, T_out + 120 K] so that a very hot agent
        (e.g. natural-gas firing) does not produce an unphysically small
        drum.

    Cost basis
    ----------
    Indirect-heat steam-tube rotary dryer, Seider, Seader, Lewin & Widagdo
    (2017), Table 16.32:

        Cp = 1,520 * A^0.80   [USD, stainless steel]

    with A = heat-transfer (tube) area [ft2]. Valid range 100-1,400 ft2;
    CE = 567, BM = 2.06. This is the same correlation already used for the
    product cooler (`SolidCooler`), which the reference gives as a combined
    rotary dryer/cooler basis.

    Notes
    -----
    The duty is built explicitly (sensible heat on the whole feed to the
    drying temperature + latent heat on the evaporated moisture) rather than
    from `H_out - H_in`. The feed stream is declared as a solid phase, so an
    enthalpy-difference duty would charge the solid->gas transition of the
    moisture (heat of sublimation, ~2,830 kJ/kg) instead of the heat of
    vaporisation of liquid water (~2,280 kJ/kg at 90 degC), overstating the
    steam demand by roughly 15-20%.
    """
    _units = {'Heat transfer area': 'ft^2',
              'Evaporation rate': 'kg/hr'}
    _N_ins = 1
    _N_outs = 2   # [0] = vapor, [1] = dried solids

    def _init(self, moisture_ID='Water', moisture_content=0.06,
              T=363.15, thermal_efficiency=0.70, U=60.0, T_medium=None):
        self.moisture_ID = moisture_ID
        self.moisture_content = moisture_content
        self.T = T
        self.thermal_efficiency = thermal_efficiency
        self.U = U
        self.T_medium = T_medium

    # -- mass balance ----------------------------------------------------
    def _run(self):
        feed = self.ins[0]
        vapor, solids = self.outs
        T_out = max(self.T, feed.T)

        solids.copy_like(feed)
        vapor.empty()
        vapor.phase = 'g'

        if solids.F_mass <= 0:
            return

        current_frac = solids.imass[self.moisture_ID] / solids.F_mass
        if current_frac <= self.moisture_content:
            # Already at or below the target - nothing to evaporate. Leave
            # the feed untouched (and cold): no drying duty is charged.
            solids.P = feed.P
            vapor.T = feed.T
            vapor.P = feed.P
            return

        vapor.copy_flow(solids, self.moisture_ID, remove=True)
        separations.adjust_moisture_content(
            solids, vapor, self.moisture_content, ID=self.moisture_ID)
        vapor.phase = 'g'
        solids.phase = feed.phase
        solids.T = vapor.T = T_out
        solids.P = vapor.P = feed.P

    # -- duty + sizing ---------------------------------------------------
    def _design(self):
        feed = self.ins[0]
        vapor, solids = self.outs

        m_evap = vapor.F_mass                       # kg/hr
        self.design_results['Evaporation rate'] = m_evap

        if m_evap <= 1e-9:
            # Pass-through (feed already dry enough): no drum, no duty.
            self.design_results['Heat transfer area'] = 0.001
            return

        T_in = feed.T
        T_out = max(self.T, T_in)

        # Sensible heat on the as-received feed + latent heat on the water
        # actually evaporated. See the class docstring for why this is not
        # taken as H_out - H_in.
        duty_sensible = feed.C * (T_out - T_in)     # kJ/hr  (feed.C: kJ/hr/K)
        chem = tmo.settings.chemicals[self.moisture_ID]
        try:
            Hvap = chem.Hvap(min(T_out, chem.Tb))   # J/mol
        except Exception:
            Hvap = chem.Hvap(chem.Tb)
        duty_latent = vapor.F_mol * Hvap            # kmol/hr * J/mol = kJ/hr
        duty = duty_sensible + duty_latent

        actual_duty = duty / self.thermal_efficiency
        self.add_heat_utility(actual_duty, T_in=T_in, T_out=T_out)

        # Tube area from Q = U * A * LMTD against the heating medium.
        T_med = self.T_medium
        if T_med is None:
            try:
                T_med = bst.settings.heating_agents[0].T
            except Exception:
                T_med = None
        if T_med is None:
            T_med = T_out + 50.0
        T_med = min(max(T_med, T_out + 10.0), T_out + 120.0)

        dT1 = T_med - T_in
        dT2 = T_med - T_out
        if abs(dT1 - dT2) < 0.01:
            LMTD = dT1
        else:
            LMTD = (dT1 - dT2) / log(dT1 / dT2)

        Q_W = abs(actual_duty) * 1000 / 3600        # kJ/hr -> W
        A_m2 = Q_W / (self.U * max(LMTD, 1.0))
        self.design_results['Heat transfer area'] = max(A_m2 * 10.7639, 0.001)


st.set_page_config(
    page_title="EXBYCost",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("EXBYCost")
st.markdown(
    "##### A fast, flexible techno-economic modeling tool for estimating "
    "extraction costs from food processing byproducts"
   )
st.markdown( "developed by the Spang lab at the University of California, Davis")
st.markdown("---")

# ── Helper to extract plain name from a feedstock chem entry (str or dict) ──
def _get_chem_name(entry):
    """Extract the plain name string from a feedstock chem entry (str or dict)."""
    if isinstance(entry, dict):
        return entry['name']
    return entry

def _fmt_mg_g(value):
    """Format a mg/g value: normal decimals when readable, engineering notation when too small."""
    if value == 0:
        return "0"
    if value >= 1:
        return f"{value:.2f}"
    if value >= 0.001:
        return f"{value:.4f}"
    # Would show as 0.0000 — switch to engineering notation
    return f"{value:.3e}"


# ============================================================================
# Sweep-mode utilities
# ============================================================================
class _NoopUI:
    """Drop-in replacement for st.empty() / st.progress() / etc.

    Used to suppress per-run Streamlit output during a parameter
    sweep so that 100 simulations don't stack 100 progress bars /
    recycle-iteration messages into the page. Every method is a
    no-op that returns self so chained calls stay valid.
    """
    def info(self, *a, **k):       pass
    def success(self, *a, **k):    pass
    def warning(self, *a, **k):    pass
    def error(self, *a, **k):      pass
    def write(self, *a, **k):      pass
    def markdown(self, *a, **k):   pass
    def caption(self, *a, **k):    pass
    def empty(self, *a, **k):      pass
    def progress(self, *a, **k):   return self
    def update(self, *a, **k):     pass
    def __enter__(self):           return self
    def __exit__(self, *a):        pass


def _fmt_duration(seconds):
    """Human-friendly duration: '12s', '1m 23s', '2h 5m'."""
    if seconds is None or not np.isfinite(seconds):
        return "—"
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"{m}m {s}s"
    h, rem = divmod(seconds, 3600)
    m, _ = divmod(rem, 60)
    return f"{h}h {m}m"


def _build_combinations(sweep_values):
    """Cartesian product of {name: [values]} → list of dicts, each
    dict mapping a name to one selected value. Order of names is
    preserved in the resulting dicts."""
    if not sweep_values:
        return [{}]
    names = list(sweep_values.keys())
    value_lists = [sweep_values[n] for n in names]
    combos = []
    for tup in itertools.product(*value_lists):
        combos.append(dict(zip(names, tup)))
    return combos


def _make_sweep_values(spec):
    """Turn a sweep spec dict for one numeric parameter into a list of
    values. spec = {'min': lo, 'max': hi, 'n': k, 'scale': 'linear'|'log',
                    'cast': type-or-None}."""
    lo = float(spec['min'])
    hi = float(spec['max'])
    n = max(2, int(spec['n']))
    if spec.get('scale', 'linear') == 'log':
        if lo <= 0 or hi <= 0:
            # Fall back to linear if log is not valid
            vals = np.linspace(lo, hi, n)
        else:
            vals = np.logspace(np.log10(lo), np.log10(hi), n)
    else:
        vals = np.linspace(lo, hi, n)
    cast = spec.get('cast')
    if cast is int:
        # Preserve order, dedupe, drop non-positive if min was >0
        vals = sorted({int(round(v)) for v in vals})
    else:
        vals = [float(v) for v in vals]
    return vals

# Define data dictionaries
feedstocks = {
    'Tomato pomace': {
        'chems': [
            {'name': 'caffeic_acid',                                         'cas': '331-39-5',            'chebi': 'CHEBI:36281'},
            {'name': 'rutin_hydrate',                                        'cas': '250249-75-3',         'chebi': 'CHEBI:232410'},
            {'name': 'luteolin',                                             'cas': '491-70-3',            'chebi': 'CHEBI:15864'},
            {'name': 'quercetin',                                            'cas': '117-39-5',            'chebi': 'CHEBI:16243'},
            {'name': 'gallic_acid',                                          'cas': '149-91-7',            'chebi': 'CHEBI:30778'},
            {'name': 'S-naringenin-1',                                      'cas': '480-41-1',            'chebi': 'CHEBI:58292'},
            {'name': 'cinnamic acid',                                        'cas': '621-82-9',            'chebi': 'CHEBI:27386'},
            {'name': '4_coumaric acid',                                      'cas': '501-98-4',            'chebi': 'CHEBI:36090'},
            {'name': 'L-alanine',                                            'cas': '56-41-7',             'chebi': 'CHEBI:16977'},
            {'name': 'aspartic acid',                                        'cas': '617-45-8',            'chebi': 'CHEBI:22660'},
            {'name': 'glutamic acid',                                        'cas': '617-65-2',            'chebi': 'CHEBI:18237'},
            {'name': 'glycine',                                              'cas': '56-40-6',             'chebi': 'CHEBI:15428'},
            {'name': 'proline',                                              'cas': '609-36-9',            'chebi': 'CHEBI:26271'},
            {'name': 'serine',                                               'cas': '302-84-1',            'chebi': 'CHEBI:17822'},
            {'name': 'arginine',                                             'cas': '7200-25-1',           'chebi': 'CHEBI:29016'},
            {'name': 'histidine',                                            'cas': '4998-57-6',           'chebi': 'CHEBI:27570'},
            {'name': 'L-isoleucine',                                         'cas': '73-32-5',             'chebi': 'CHEBI:17191'},
            {'name': 'leucine',                                              'cas': '328-39-2',            'chebi': 'CHEBI:25017'},
            {'name': 'L-lysine',                                             'cas': '56-87-1',             'chebi': 'CHEBI:18019'},
            {'name': 'threonine',                                            'cas': '80-68-2',             'chebi': 'CHEBI:26986'},
            {'name': 'D-tyrosine',                                           'cas': '0556-02-05',          'chebi': 'CHEBI:28479'},
            {'name': 'valine',                                               'cas': '0516-06-03',          'chebi': 'CHEBI:27266'},
            {'name': 'lycopene',                                             'cas': '502-65-8',            'chebi': 'CHEBI:15948'},
            {'name': 'beta-carotene',                                        'cas': '7235-40-7',           'chebi': 'CHEBI:17579'},
            {'name': 'lutein',                                               'cas': '127-40-2',            'chebi': 'CHEBI:28838'},
            {'name': 'isofucosterol',                                        'cas': '18472-36-1',          'chebi': 'CHEBI:28604'},
            {'name': 'campesterol',                                          'cas': '474-62-4',            'chebi': 'CHEBI:28623'},
            {'name': 'stigmasterol',                                         'cas': '83-48-7',             'chebi': 'CHEBI:28824'},
            {'name': 'sitosterol',                                           'cas': '83-46-5',             'chebi': 'CHEBI:27693'},
            {'name': '5alpha-cholestan-3-beta-ol',                           'cas': '80-97-7',             'chebi': 'CHEBI:86570'},
            {'name': '5alpha-cholesten-7-3-beta-ol',                         'cas': '80-99-9',             'chebi': 'CHEBI:17168'},
            {'name': '24,25-dihydrolanosterol',                              'cas': '79-62-9',             'chebi': 'CHEBI:28113'},
            {'name': 'cholesterol',                                          'cas': '57-88-5',             'chebi': 'CHEBI:16113'},
            {'name': '2,4-Oxocholesterol',                                    'cas': '17752-16-8',          'chebi': 'CHEBI:166803'},
            {'name': 'S,R,R,alpha tocopherol',                              'cas': '59-02-9',             'chebi': 'CHEBI:177086'},
            {'name': 'delta tocopherol',                                     'cas': '119-13-1',            'chebi': 'CHEBI:47772'},
            {'name': 'squalene',                                             'cas': '0111-02-04',          'chebi': 'CHEBI:15440'},
            {'name': 'cycloartenol',                                         'cas': '469-38-5',            'chebi': 'CHEBI:17030'},
            {'name': '2 cis,6 trans-farnesol',                              'cas': '3790-71-4',           'chebi': 'CHEBI:16774'},
            {'name': 'beta amyrin',                                          'cas': '559-70-6',            'chebi': 'CHEBI:10352'},
            {'name': 'oleanolic acid',                                       'cas': '0508-02-01',          'chebi': 'CHEBI:37659'},
            {'name': 'ursolic acid',                                         'cas': '77-52-1',             'chebi': 'CHEBI:9908'},
            {'name': 'phloretic acid',                                       'cas': '501-97-3',            'chebi': 'CHEBI:32980'},
            {'name': 'ferulic acid',                                         'cas': '1135-24-6',           'chebi': 'CHEBI:193350'},
            {'name': 'sinapic acid',                                         'cas': '530-59-6',            'chebi': 'CHEBI:77131'},
            {'name': 'chlorogenic acid',                                     'cas': '327-97-9',            'chebi': 'CHEBI:16112'},
            {'name': '4,hydroxybenzoic acid',                                'cas': '99-96-7',             'chebi': 'CHEBI:30763'},
            {'name': 'vanillic acid',                                        'cas': '121-34-6',            'chebi': 'CHEBI:30816'},
            {'name': 'syringic acid',                                        'cas': '530-57-4',            'chebi': 'CHEBI:68329'},
            {'name': 'chrysin',                                              'cas': '480-40-0',            'chebi': 'CHEBI:75095'},
            {'name': 'epicatechin',                                          'cas': '490-46-0',            'chebi': 'CHEBI:90'},
            {'name': 'catechin',                                             'cas': '154-23-4',            'chebi': 'CHEBI:15600'},
            {'name': 'kaempferol',                                           'cas': '520-18-3',            'chebi': 'CHEBI:28499'},
            {'name': 'resveratrol',                                          'cas': '501-36-0',            'chebi': 'CHEBI:27881'},
            {'name': 'hexadecanoic acid',                                    'cas': '57-10-3',             'chebi': 'CHEBI:15756'},
            {'name': 'octadecanoic acid',                                    'cas': '57-11-4',             'chebi': 'CHEBI:28842'},
            {'name': 'oleic acid',                                           'cas': '112-80-1',            'chebi': 'CHEBI:16196'},
            {'name': 'linoleic acid',                                        'cas': '60-33-3',             'chebi': 'CHEBI:17351'},
            {'name': 'alpha-linolenic acid',                                 'cas': '463-40-1',            'chebi': 'CHEBI:27432'},
            {'name': 'beta-tocopherol',                                      'cas': '0148-03-08',          'chebi': 'CHEBI:47771'},
            {'name': 'gamma-tocopherol',                                     'cas': '54-28-4',             'chebi': 'CHEBI:18185'},
            {'name': 'zeaxanthin',                                           'cas': '144-68-3',            'chebi': 'CHEBI:27547'},
            {'name': 'delta-carotene',                                       'cas': '472-92-4',            'chebi': 'CHEBI:27705'},
            {'name': 'rutin',                                                'cas': '153-18-4',            'chebi': 'CHEBI:28527'},
            {'name': 'lignin',                                               'cas': '9005-53-2',           'chebi': 'CHEBI:6457'},
            {'name': 'cystine',                                              'cas': '923-32-0',            'chebi': 'CHEBI:17376'},
            {'name': 'methionine',                                           'cas': '59-51-8',             'chebi': 'CHEBI:16811'},
            {'name': 'phenylalanine',                                        'cas': '150-30-1',            'chebi': 'CHEBI:28044'},
            {'name': 'catechin hydrate',                                     'cas': '88191-48-4',          'chebi': None},
            {'name': '15-cis-neurosporene',                                  'cas': '502-64-7',            'chebi': None},
            {'name': 'Cellulose',                                            'cas': '9004-34-6',           'chebi': 'CHEBI:18246'},
        ],
        'comp': [5.378e-05, 0.00032512, 7.3e-07, 3.47e-05, 8.198e-05, 0.0001641525, 1.8e-06, 7.693333333e-06, 0.008630384, 0.028214528, 0.04969716, 0.015757848, 0.011231288, 0.010079424, 0.0301307, 0.010293092, 0.011505852, 0.020030436, 0.02148212, 0.01158646, 0.011176904, 0.013830928, 0.0002737587415, 0.0002384133333, 1.505e-05, 6.23e-05, 6.56e-05, 0.0001517, 0.0003788, 9.7e-06, 3.6e-06, 5.24e-05, 4.19e-05, 6.75e-05, 0.0001557, 2.3e-07, 9.3e-06, 5.18e-05, 3.2e-06, 0.0001778, 1.25e-05, 5.76e-05, 4.1e-06, 1.653e-05, 7.2e-06, 5.17e-05, 1.8e-06, 4.1e-06, 2.2915e-05, 2.95e-05, 2.2e-06, 4.4e-06, 5.5e-06, 7.7e-06, 0.0318671, 0.0115962, 0.0511363, 0.082882, 0.002697, 3.24e-06, 0.00011665, 3.6e-07, 1e-05, 0.00048926, 0.254, 0.0078, 0.00897, 0.02262, 0.00049112, 2.059e-05, 0.2689966031],
        'moisture': 0.67,
        'type': 'Fruit pomaces'
    },
    'Almond hulls': {
        'chems': [
            {'name': '3,4 dihydroxybenzoic acid',                            'cas': '99-50-3',             'chebi': 'CHEBI:36062'},
            {'name': 'chlorogenic acid',                                     'cas': '327-97-9',            'chebi': 'CHEBI:16112'},
            {'name': 'naringenin-7-O-beta-D-glucoside',                      'cas': '529-55-5',            'chebi': 'CHEBI:28327'},
            {'name': 'kaempferol-3-rutinoside',                              'cas': '17650-84-9',          'chebi': 'CHEBI:69657'},
            {'name': 'Kaempferol-3-O-glucoside',                            'cas': '0480-10-4',           'chebi': 'CHEBI:30200'},
            {'name': 'isorhamnetin-3-O-rutinoside',                          'cas': '604-80-8',            'chebi': 'CHEBI:145096'},
            {'name': 'isorhamnetin-3-O-beta-D-glucopyranoside',              'cas': '5041-82-7',           'chebi': 'CHEBI:75750'},
            {'name': 'isorhamnetin',                                         'cas': '480-19-3',            'chebi': 'CHEBI:6052'},
            {'name': '4-coumaric acid',                                      'cas': '501-98-4',            'chebi': 'CHEBI:36090'},
            {'name': 'vanillic acid',                                        'cas': '121-34-6',            'chebi': 'CHEBI:30816'},
            {'name': 'lignin',                                               'cas': '9005-53-2',           'chebi': 'CHEBI:6457'},
            {'name': 'hemicellulose',                                        'cas': '9034-32-6',           'chebi': 'CHEBI:61266'},
            {'name': 'starch',                                               'cas': '9005-25-8',           'chebi': 'CHEBI:28017'},
            {'name': 'Cellulose',                                            'cas': '9004-34-6',           'chebi': 'CHEBI:18246'},
        ],
        'comp': [2.545e-06, 1.4655e-05, 2.6435e-05, 1.06e-06, 6.55e-07, 4.185e-06, 7.4e-07, 1.095e-06, 5.67e-06, 1.7e-06, 0.0925, 0.021, 0.026, 0.740438815],
        'moisture': 0.135,
        'type': 'hulls'
    },
    'Pistachio hulls': {
        'chems': [
            {'name': 'gallic acid',                                          'cas': '149-91-7',            'chebi': 'CHEBI:30778'},
            {'name': 'catechin',                                             'cas': '154-23-4',            'chebi': 'CHEBI:15600'},
            {'name': 'epicatechin',                                          'cas': '490-46-0',            'chebi': 'CHEBI:90'},
            {'name': 'eriodictyol-7-O-beta-D-glucopyranoside',               'cas': '38965-51-4',          'chebi': 'CHEBI:139458'},
            {'name': 'naringin',                                             'cas': '10236-47-2',          'chebi': 'CHEBI:28819'},
            {'name': 'eriodictyol',                                          'cas': '552-58-9',            'chebi': 'CHEBI:28412'},
            {'name': 'quercetin',                                            'cas': '117-39-5',            'chebi': 'CHEBI:16243'},
            {'name': 'S-naringenin-1',                                      'cas': '480-41-1',            'chebi': 'CHEBI:58292'},
            {'name': 'luteolin',                                             'cas': '491-70-3',            'chebi': 'CHEBI:15864'},
            {'name': 'kaempferol',                                           'cas': '520-18-3',            'chebi': 'CHEBI:28499'},
            {'name': '4-hydroxybenzoic acid',                                'cas': '99-96-7',             'chebi': 'CHEBI:30763'},
            {'name': '3,4,dihydroxybenzoic acid',                            'cas': '99-50-3',             'chebi': 'CHEBI:36062'},
            {'name': 'syringic acid',                                        'cas': '530-57-4',            'chebi': 'CHEBI:68329'},
            {'name': '4,coumaric acid',                                      'cas': '501-98-4',            'chebi': 'CHEBI:36090'},
            {'name': 'caffeic acid',                                         'cas': '331-39-5',            'chebi': 'CHEBI:36281'},
            {'name': 'cyanidin-3-O-beta-D-glucoside',                        'cas': '7084-24-4',           'chebi': 'CHEBI:28426'},
            {'name': 'Cyanidin-3-O-galactoside',                             'cas': '27661-36-5',          'chebi': None},
            {'name': 'Cellulose',                                            'cas': '9004-34-6',           'chebi': 'CHEBI:18246'},
        ],
        'comp': [0.001609565385, 9.354166668e-05, 5.508384616e-05, 0.0001270833333, 2.715833334e-05, 1.719166667e-05, 6.708666668e-06, 3.325e-06, 4.350000001e-06, 8.000000001e-07, 0.0020625, 0.00031614, 1.914e-05, 1.716e-05, 1.188e-05, 3.882262211e-07, 0.0014306075, 0.9941973764],
        'moisture': 0.740667,
        'type': 'hulls'
    },
    'Pomegranate Peel': {
        'chems': [
            {'name': 'gallic acid',                                          'cas': '149-91-7',            'chebi': 'CHEBI:30778'},
            {'name': 'punicalin 2',                                          'cas': '65995-64-4',          'chebi': 'CHEBI:234446'},
            {'name': 'alpha-punicalagin',                                     'cas': '130518-17-1',         'chebi': 'CHEBI:233620'},
            {'name': 'beta-punicalagin',                                     'cas': '130608-10-5',         'chebi': 'CHEBI:233621'},
            {'name': 'ellagic acid',                                         'cas': '476-66-4',            'chebi': 'CHEBI:4775'},
            {'name': 'cyanin',                                               'cas': '20905-74-2',          'chebi': 'CHEBI:3978'},
            {'name': 'cyanidin-3-O-beta-D-glucoside',                        'cas': '7084-24-4',           'chebi': 'CHEBI:28426'},
            {'name': 'pelargonin',                                           'cas': '17334-58-6',          'chebi': 'CHEBI:133365'},
            {'name': 'pelargonidin-3-O-beta-D-glucoside',                    'cas': '18466-51-8',          'chebi': 'CHEBI:31967'},
            {'name': 'catechin',                                             'cas': '154-23-4',            'chebi': 'CHEBI:15600'},
            {'name': 'epicatechin',                                          'cas': '490-46-0',            'chebi': 'CHEBI:90'},
            {'name': 'kaempferol-O-glucoside',                              'cas': '480-10-4',           'chebi': 'CHEBI:30200'},
            {'name': 'Ellagic acid pentoside',                               'cas': '139163-18-1',         'chebi': 'CHEBI:167700'},
            {'name': 'corilagin',                                            'cas': '23094-69-1',          'chebi': 'CHEBI:3884'},
            {'name': 'Pedunculagin',                                         'cas': '7045-42-3',           'chebi': 'CHEBI:7948'},
            {'name': 'Peganine',                                             'cas': '6159-55-3',           'chebi': 'CHEBI:7949'},
            {'name': 'Casuarinin',                                           'cas': '79786-01-09',         'chebi': 'CHEBI:3462'},
            {'name': 'Granatin B',                                           'cas': '77322-54-4',          'chebi': 'CHEBI:167697'},
            {'name': 'punicalagin',                                          'cas': '65995-63-3',          'chebi': 'CHEBI:167695'},
            {'name': 'procyanidin B1',                                       'cas': '82262-99-5',          'chebi': 'CHEBI:75633'},
            {'name': 'proanthocyanidin',                                     'cas': None,                  'chebi': 'CHEBI:26267'},
            {'name': '3,4,dihydroxybenzoic_acid',                            'cas': '99-50-3',             'chebi': 'CHEBI:36062'},
            {'name': 'trans-4-coumaric acid',                                'cas': '501-98-4',            'chebi': 'CHEBI:32374'},
            {'name': 'phlorizin',                                            'cas': '60-81-1',             'chebi': 'CHEBI:8113'},
            {'name': 'epicatechin gallate',                                 'cas': '1257-08-05',          'chebi': 'CHEBI:70255'},
            {'name': '1,2,6,tris-O-galloyl-beta-D-glucose',                  'cas': '79886-49-0',          'chebi': 'CHEBI:27395'},
            {'name': 'procyanidin B2',                                       'cas': '29106-49-8',          'chebi': 'CHEBI:75632'},
            {'name': 'procyanidin B3',                                       'cas': '12798-58-2',          'chebi': 'CHEBI:75619'},
            {'name': 'vanillic acid',                                        'cas': '121-34-6',            'chebi': 'CHEBI:30816'},
            {'name': 'Brevifolincarboxylic acid',                            'cas': '18490-95-4',          'chebi': 'CHEBI:228853'},
            {'name': 'kaempferol-3-rutinoside',                              'cas': '17650-84-9',          'chebi': 'CHEBI:69657'},
            {'name': 'vanillic acid hexoside',                               'cas': None,                  'chebi': None},
            {'name': 'Valoneic acid dilactone',                              'cas': None,                  'chebi': None},
            {'name': 'cis dihydrokaempferol hexoside',                       'cas': '1574305-24-0',        'chebi': None},
            {'name': 'hydroxycaffeic acid',                                  'cas': '56225-67-3',          'chebi': None},
            {'name': 'Galloylglucose',                                       'cas': '13186-19-1',          'chebi': None},
            {'name': 'Ellagic acid rhamnoside',                              'cas': None,                  'chebi': None},
            {'name': 'Cellulose',                                            'cas': '9004-34-6',           'chebi': 'CHEBI:18246'},
        ],
        'comp': [2.9e-05, 3.94e-04, 2.905e-03, 3.163e-03, 1.25e-04, 3.1e-05, 2.7e-05, 9e-06, 1.5e-05, 1e-06, 0, 4e-06, 9e-06, 0, 3.9e-05, 0, 0, 3.9e-05, 2.9e-04, 0, 0, 1e-06, 1e-06, 0, 0, 0, 0, 0, 1e-06, 3e-06, 7e-06, 0, 0, 0, 1e-06, 4.1e-05, 1.1e-05, 0.992852],
        'moisture': 0.156667,
        'type': 'hulls'
    },
}

cepci_index = {
    2010: 550.8, 2011: 585.7, 2012: 584.6, 2013: 567.3, 2014: 576.1,
    2015: 556.8, 2016: 541.7, 2017: 567.5, 2018: 603.1, 2019: 607.5,
    2020: 596.2, 2021: 708.8, 2022: 816.0, 2023: 797.9, 2024: 799.0,
    2025: 798.88,
    # Preliminary 2026 estimate. The final 2026 annual average is not yet
    # published; early-2026 monthly CEPCI readings were running ~7.5–8%
    # above the year-earlier values amid continued inflationary pressure
    # (Chemical Engineering, 2026). Update this once the annual average is
    # released.
    2026: 825.0,
}

# ── Solvent prices ──────────────────────────────────────────────────
# Base prices, base years, regions and sources live in
# solvent_prices.csv (keep it next to this script). If a FRED API key is
# present in the environment (FRED_API_KEY), base prices are escalated
# to the present using BLS Producer Price Indices; otherwise the cited
# base prices are used as-is. Free key: fred.stlouisfed.org/docs/api
try:
    _FRED_API_KEY = st.secrets.get('FRED_API_KEY') or os.environ.get('FRED_API_KEY')
except (FileNotFoundError, KeyError):
    _FRED_API_KEY = os.environ.get('FRED_API_KEY')

@st.cache_data(show_spinner=False, ttl=86400)   # 24-h TTL: daily refresh
def _load_solvent_prices(api_key, escalate):
    # cached so FRED is queried at most once per day per session
    return get_solvent_prices(fred_api_key=api_key, escalate=escalate)


@st.cache_data(show_spinner=False, ttl=86400)   # 24-h TTL: daily refresh
def _load_california_electricity_price(api_key):
    # cached so EIA is queried at most once per day per session
    return get_california_industrial_price(api_key)


@st.cache_data(show_spinner=False, ttl=86400)   # 24-h TTL: daily refresh
def _load_california_operator_wage(bls_key, fred_key):
    # cached so BLS+FRED are queried at most once per day per session
    return get_california_operator_wage(bls_key, fred_api_key=fred_key)


try:
    solvprice_index, solvprice_meta = _load_solvent_prices(
        _FRED_API_KEY, bool(_FRED_API_KEY))
except Exception as _price_err:
    st.error(f"solvent_prices.csv could not be loaded: {_price_err}. "
             f"Place solvent_prices.csv next to this script.")
    st.stop()

porosity_torosity = {
    'Fresh fruit and veg': 0.132,
    'Fruit pomaces': 0.423,
    'nuts, seeds and grains': 0.313,
    'hulls': 0.476,
    'stalks and straw': 0.483,
    'leaves': 0.797,
    'woody biomass': 0.225
}

depreciation_options = [
    'MACRS3', 'MACRS5', 'MACRS7', 'MACRS10', 'MACRS15', 'MACRS20',
    'SL5', 'SL7', 'SL10', 'SL15', 'SL20',
    'DDB5', 'DDB7', 'DDB10', 'DDB15', 'DDB20',
    'SYD5', 'SYD7', 'SYD10', 'SYD15', 'SYD20',
]
#heating utilities
HeatU=['low_pressure_steam','medium_pressure_steam','high_pressure_steam','natural_gas']

# =============================================================================
# Default TEA values — two profiles
# =============================================================================
TEA_DEFAULTS_NTH = {
    'IRR': 0.10,
    'duration': 30,
    'operating_days': 329,
    'depreciation': 'MACRS7',
    'income_tax': 0.21,
    'construction_schedule': (0.08, 0.60, 0.32),
    'startup_months': 6,
    'startup_FOCfrac': 1.0,
    'startup_VOCfrac': 0.75,
    'startup_salesfrac': 0.5,
    'finance_interest': 0.08,
    'finance_years': 10,
    'finance_fraction': 0.40,
    'WC_over_FCI': 0.05,
    'maintenance': 0.03,
    'property_insurance': 0.007,
    'property_tax': 0.01,
    'Additional_direct_costs': 0.175,
    'indirect_costs': 0.60,
    'operator_hourly_wage': 23.45,
    'supervision_factor': 0.15,
    'fringe_benefits': 0.40,
    'supplies': 0.10,
    'administration': 0.20,
    'elec_price': 0.19,
    'feed_price': 0.0,
    'feed_storage_days': 5,
    'solv_storage_days': 5,
}

TEA_DEFAULTS_FOAK = {
    'IRR': 0.15,
    'duration': 25,
    'operating_days': 310,
    'depreciation': 'MACRS7',
    'income_tax': 0.21,
    'construction_schedule': (0.08, 0.30, 0.30, 0.32),
    'startup_months': 21,
    'startup_FOCfrac': 1.25,
    'startup_VOCfrac': 0.85,
    'startup_salesfrac': 0.4,
    'finance_interest': 0.10,
    'finance_years': 12,
    'finance_fraction': 0.50,
    'WC_over_FCI': 0.07,
    'maintenance': 0.03,
    'property_insurance': 0.007,
    'property_tax': 0.01,
    'Additional_direct_costs': 0.175,
    'indirect_costs': 0.60,
    'operator_hourly_wage': 23.45,
    'supervision_factor': 0.15,
    'fringe_benefits': 0.40,
    'supplies': 0.10,
    'administration': 0.20,
    'elec_price': 0.19,
    'feed_price': 0.0,
    'feed_storage_days': 5,
    'solv_storage_days': 5,
}

# Keys that the 'TEA Parameters' dropdown (Nth Plant / FOAK) controls --
# i.e. exactly the assumptions the interactive Nth/FOAK block assigns. The
# operator wage, electricity price, feed price and storage days are set
# independently in the sidebar (from their own data sources / inputs) and
# must NOT be overridden when a reactor choice forces FOAK in a sweep.
_TEA_SIDEBAR_KEYS = {'operator_hourly_wage', 'elec_price', 'feed_price',
                     'feed_storage_days', 'solv_storage_days'}
TEA_MODE_KEYS = frozenset(TEA_DEFAULTS_FOAK) - _TEA_SIDEBAR_KEYS

# Sidebar - Input Parameters
st.sidebar.header("📊 Input Parameters")

# =========================================================================
# 0. Mode
# =========================================================================
sim_mode = st.sidebar.radio(
    "Mode",
    options=["Single Run", "Parameter Sweep"],
    horizontal=True,
    help=("Single Run: solve one set of parameters and view the full TEA "
          "report. Parameter Sweep: choose one or more parameters to vary "
          "over a range, run them all, and download the combined results "
          "as a CSV."),
)
st.sidebar.markdown("---")

# =========================================================================
# 1. Feedstock Selection
# =========================================================================
st.sidebar.subheader("1. Feedstock Selection")
selected_feedstock = st.sidebar.selectbox(
    "Select Feedstock",
    options=list(feedstocks.keys()),
    help="Choose the biomass feedstock type for extraction.  Biomass composition found from the food byproduct database, any missing weight assumed to be cellulose. https://www.byproductdatabase.com/"
)

# (Full feedstock composition shown in main area expander)

# =========================================================================
# 2. Process Settings
# =========================================================================
st.sidebar.subheader("2. Process Settings")
feedflow = st.sidebar.number_input(
    "Feed Flow Rate (kg/hr)",
    min_value=10.0, max_value=1000000.0, value=400.0, step=10.0,
    help="Mass flow rate of feedstock entering the system."
)

solv = st.sidebar.selectbox(
    "Solvent Type",
    options=list(solvprice_index.keys()),
    help="Extraction solvent used to separate target compounds from the feedstock."
)

if solv in solvprice_index:
    st.sidebar.caption(f"Using **${solvprice_index[solv]:.4f}/kg**")

solv_flow_mode = st.sidebar.radio(
    "Solvent Flow Specification",
    options=["Exact flow (kg/hr)", "Solvent/feed ratio"],
    help="Specify the solvent feed either directly as a mass flow rate, or as "
         "a mass ratio of solvent to feed. In ratio mode the solvent flow is "
         "computed as ratio × feed flow (so it tracks the feed flow, including "
         "when feed flow is varied in a sweep)."
)

if solv_flow_mode == "Exact flow (kg/hr)":
    solvflow = st.sidebar.number_input(
        "Solvent Flow Rate (kg/hr)",
        min_value=10.0, max_value=1000000.0, value=1000.0, step=10.0,
        help="Mass flow rate of solvent fed to the extractor."
    )
    solv_feed_ratio = (solvflow / feedflow) if feedflow else 0.0
    st.sidebar.caption(
        f"→ Solvent/feed ratio = **{solv_feed_ratio:.3g} kg/kg**"
    )
else:
    solv_feed_ratio = st.sidebar.number_input(
        "Solvent/Feed Ratio (kg/kg)",
        min_value=0.01, max_value=1000.0, value=2.5, step=0.1,
        help="Mass ratio of solvent to feed. The solvent flow rate is computed "
             "as this ratio × feed flow rate."
    )
    solvflow = solv_feed_ratio * feedflow
    st.sidebar.caption(
        f"→ Solvent flow = **{solvflow:,.2f} kg/hr** "
        f"({solv_feed_ratio:.3g} × {feedflow:,.0f} kg/hr feed)"
    )

ExtractT = st.sidebar.number_input(
    "Solvent Temperature (°C)",
    min_value=0.0, max_value=100.0, value=35.0, step=1.0,
    help="Operating temperature of the solvent entering the extractor. "
)

extractt = st.sidebar.number_input(
    "Extraction Time (hr)",
    min_value=0.1, max_value=24.0, value=2.0, step=0.1,
    help="Residence time of feedstock in the extractor."
)

Particlesize = st.sidebar.number_input(
    "Particle size (cm)",
    min_value=0.01, max_value=5.0, value=0.25, step=0.01,
    help="Radius of feedstock particles after grinding (cm)."
)

reactor_type = st.sidebar.selectbox(
    "Reactor Type",
    options=['conventional', 'ultrasound', 'microwave'],
    help="Type of extraction reactor"

)

heatutility = st.sidebar.selectbox(
    "Heat Utility",
    options=HeatU,
    help="Choose the method for supplying heat utility"
)

pressure_mode = st.sidebar.selectbox(
    "Pressure Mode",
    options=["Calculated", "Custom"],
    help="Calculated: Automatically sets pressure to 5% above the solvent's vapor pressure at extraction temperature (ensures solvent remains liquid). Custom: Manually specify operating pressure."
)

if pressure_mode == "Custom":
    ExtractP_custom = st.sidebar.number_input(
        "Extraction Pressure (Pa)",
        min_value=1000.0, max_value=1000000.0, value=101325.0, step=1000.0,
        help="Operating pressure in Pascals."
    )
else:
    ExtractP_custom = None

# =========================================================================
# 2a. Feedstock Drying (pre-storage)
# =========================================================================
st.sidebar.subheader("2a. Feedstock Drying")

_as_received_moisture = feedstocks[selected_feedstock]['moisture']
st.sidebar.caption(
    f"As-received moisture: **{_as_received_moisture * 100:.1f} wt%**"
)

feed_dry_enable = st.sidebar.checkbox(
    "Dry Feedstock Before Storage",
    value=True,
    help=("Rotary drum dryer on the as-received feedstock, upstream of the "
          "storage tank. Drying before storage prevents mould, self-heating "
          "and loss of extractables in the pile, and stops feed moisture "
          "being carried through extraction into the solvent loop and the "
          "final product.")
)

FEED_DRY_STABLE_STORAGE = 0.06  # 6 wt% — stable long-term storage target

# A dryer can only take water OUT, so the target moisture is bounded above
# by the moisture that actually arrives with the selected feedstock. The
# slider/box limits therefore track the feedstock, not a fixed 60%.
_max_moisture_pct = max(round(_as_received_moisture * 100, 1), 0.2)
_min_moisture_pct = min(0.5, _max_moisture_pct / 2.0)
_default_moisture_pct = min(max(round(FEED_DRY_STABLE_STORAGE * 100, 1),
                                _min_moisture_pct), _max_moisture_pct)

feed_dry_spec = st.sidebar.radio(
    "Target Feed Moisture",
    options=["6% — stable long-term storage", "Custom"],
    help=("6 wt% (wet basis) is the standard target for stable long-term "
          "storage of dried biomass — below roughly 10 wt% microbial and "
          "enzymatic activity is arrested, and 6 wt% leaves margin for "
          "moisture pick-up in the tank. Choose Custom to set your own "
          "target (e.g. a milder dry if the compounds of interest are "
          "heat-sensitive, or a harder dry to keep more water out of the "
          "solvent loop). The target is capped at the as-received moisture "
          "of the selected feedstock — the dryer cannot add water."),
    disabled=not feed_dry_enable,
)

if feed_dry_spec == "Custom":
    feed_dry_moisture = st.sidebar.number_input(
        "Custom Feed Moisture (wt%)",
        min_value=float(_min_moisture_pct),
        max_value=float(_max_moisture_pct),
        value=float(_default_moisture_pct), step=0.5,
        format="%.1f",
        help=(f"Moisture in the dried feedstock leaving the drum dryer, "
              f"wet basis (kg water / kg wet solids). Upper limit is the "
              f"as-received moisture of "
              f"{selected_feedstock} ({_max_moisture_pct:.1f} wt%) — at "
              f"that value the dryer is a pass-through."),
        disabled=not feed_dry_enable,
    ) / 100.0
else:
    # Never target a moisture the feed does not have.
    feed_dry_moisture = min(FEED_DRY_STABLE_STORAGE, _as_received_moisture)

feed_dry_T = st.sidebar.number_input(
    "Feed Drying Temperature (°C)",
    min_value=40.0, max_value=200.0, value=90.0, step=5.0,
    help="Solids outlet temperature in the drum dryer. Higher is faster and "
         "needs less heat-transfer area, but thermolabile target compounds "
         "are flagged against this temperature in the degradation check.",
    disabled=not feed_dry_enable,
)

if feed_dry_enable:
    if feed_dry_moisture >= _as_received_moisture:
        st.sidebar.info(
            f"Target ({feed_dry_moisture * 100:.1f}%) is at or above the "
            f"as-received moisture ({_as_received_moisture * 100:.1f}%) — "
            f"the dryer will pass the feed through with no evaporation."
        )
    else:
        # Wet-basis balance: dry solids are conserved.
        _dry_solids = feedflow * (1.0 - _as_received_moisture)
        _dried_feed = _dry_solids / (1.0 - feed_dry_moisture)
        _water_out = feedflow - _dried_feed
        st.sidebar.caption(
            f"→ Evaporates **{_water_out:,.1f} kg/hr** water; "
            f"**{_dried_feed:,.1f} kg/hr** to storage"
        )

# Dried feed is cooled to ambient before it enters the storage tank —
# storing hot dried biomass causes condensation on the tank walls and
# re-wetting of the pile.
FEED_COOL_T = 298.15  # K (25 °C)

# =========================================================================
# 2b. Post-Extraction Concentration Settings
# =========================================================================
st.sidebar.subheader("2b. Concentration & Drying")
evap_n_effects = st.sidebar.number_input(
    "Evaporator Effects",
    min_value=1, max_value=5, value=3, step=1,
    help="Number of effects in the multi-effect evaporator. More effects = less steam but higher capital cost."
)
evap_target_solids = st.sidebar.slider(
    "Evaporator Target Solids (wt%)",
    min_value=10, max_value=50, value=45, step=5,
    help="Target solids fraction leaving the evaporator. Max 50% due to viscosity limits [Tanguy et al. 2015]."
) / 100.0
dryer_moisture = st.sidebar.slider(
    "Dryer Residual Solvent (wt%)",
    min_value=0.1, max_value=20.0, value=5.0, step=0.1, format="%.1f",
    help="Residual solvent in the final dried product."
) / 100.0

# Cooling settings (hardcoded — no user input needed)
solid_cooler_U = 25  # W/(m²·K), typical for indirect rotary coolers
                     # with organic powders (Nhuchhen et al. 2016; Perry's §11)

# Feedstock drum-dryer settings (hardcoded — no user input needed)
feed_dryer_efficiency = 0.70  # fraction of supplied heat reaching the solids;
                              # typical for indirect steam-tube rotary dryers
                              # (Mujumdar, Handbook of Industrial Drying, Ch. 2)
feed_dryer_U = 60             # W/(m²·K) on the tube area, steam-tube rotary
                              # dryer on wet granular solids (Perry's §12)

# =========================================================================
# 2c. Solvent Recycle
# =========================================================================
st.sidebar.subheader("2c. Solvent Recycle")
enable_recycle = st.sidebar.checkbox(
    "Enable Solvent Recycle",
    value=True,
    help=("Recover solvent from the evaporator condensate and the "
          "(condensed) dryer vapor, mix with fresh makeup, and feed "
          "back to the extractor. A purge stream prevents impurity "
          "buildup in the loop.")
)
recycle_x_max = st.sidebar.number_input(
    "Max Impurity Mass Percent at Solvent Inlet (%)",
    min_value=0.1, max_value=20.0, value=2.0, step=0.5,
    format="%.1f",
    help=("Cap on the mass percent of non-solvent impurities "
          "(everything except the chosen solvent — water, volatile "
          "extractables, etc.) in the combined solvent stream entering "
          "the extractor heater. The recycle/purge split is solved by "
          "bisection to satisfy this cap."),
    disabled=not enable_recycle,
) / 100.0
# =========================================================================
# 3. TEA Settings
# =========================================================================
st.sidebar.subheader("3. TEA Settings")

Yearstart = st.sidebar.selectbox(
    "Start Year",
    options=list(cepci_index.keys()),
    index=list(cepci_index.keys()).index(2026),
    help="Year when the facility begins operation. Determines the CEPCI used for equipment cost estimation."
)

col1, col2 = st.sidebar.columns(2)
with col1:
    elec_source = st.radio(
        "Electricity price source",
        options=("California (EIA, live)", "Custom"),
        help="California pulls the latest California industrial retail "
             "electricity price from EIA (requires the EIA_API_KEY env "
             "var; free key at eia.gov/opendata/register.php). Custom "
             "lets you enter your own $/kWh."
    )
    if elec_source.startswith("California"):
        try:
            try:
                _eia_key = st.secrets.get('EIA_API_KEY') or os.environ.get('EIA_API_KEY')
            except (FileNotFoundError, KeyError):
                _eia_key = os.environ.get('EIA_API_KEY')
            elec_price, _elec_meta = _load_california_electricity_price(_eia_key)
            st.caption(f"Using **${elec_price:.4f}/kWh** "
                       f"({_elec_meta['period']}, EIA CA Industrial)")
        except Exception as _elec_err:
            st.error(f"EIA electricity price unavailable: {_elec_err}")
            st.stop()
    else:
        elec_price = st.number_input(
            "Electricity Price ($/kWh)",
            value=TEA_DEFAULTS_NTH['elec_price'], format="%.4f",
            help="Cost of electricity for operating pumps, motors, and other electrical equipment."
        )
    feed_price = st.number_input(
        "Feed Price ($/kg)",
        value=TEA_DEFAULTS_NTH['feed_price'], format="%.4f",
        help="Purchase cost of feedstock per kg."
    )
with col2:
    feed_storage_days = st.number_input(
        "Feed Storage (days)",
        min_value=1, max_value=30, value=TEA_DEFAULTS_NTH['feed_storage_days'],
        help="Days of feedstock inventory to maintain on-site."
    )
    solv_storage_days = st.number_input(
        "Solvent Storage (days)",
        min_value=1, max_value=30, value=TEA_DEFAULTS_NTH['solv_storage_days'],
        help="Days of solvent inventory to maintain on-site."
    )

wage_source = st.sidebar.radio(
    "Operator wage source",
    options=("California (BLS OEWS, live)", "Custom"),
    help="California pulls the latest California state hourly mean wage "
         "for SOC 51-8091 (Chemical Plant and System Operators) from the "
         "BLS Public Data API (requires the BLS_API_KEY env var; free key "
         "at data.bls.gov/registrationEngine/). Custom lets you enter "
         "your own $/hr. Note: OEWS is annual, refreshed once per year."
)
if wage_source.startswith("California"):
    try:
        try:
            _bls_key = st.secrets.get('BLS_API_KEY') or os.environ.get('BLS_API_KEY')
            _fred_key = st.secrets.get('FRED_API_KEY') or os.environ.get('FRED_API_KEY')
        except (FileNotFoundError, KeyError):
            _bls_key = os.environ.get('BLS_API_KEY')
            _fred_key = os.environ.get('FRED_API_KEY')
        operator_hourly_wage, _wage_meta = _load_california_operator_wage(
            _bls_key, _fred_key)
        if _wage_meta.get('escalated'):
            _period_label = (f"May {_wage_meta['period']} OEWS escalated "
                             f"to {_wage_meta['eci_latest_date']} via ECI")
        else:
            _period_label = (f"May {_wage_meta['period']}, "
                             f"BLS OEWS SOC 51-8091")
        st.sidebar.caption(f"Using **${operator_hourly_wage:.2f}/hr** "
                           f"({_period_label})")
    except Exception as _wage_err:
        st.error(f"BLS operator wage unavailable: {_wage_err}")
        st.stop()
else:
    operator_hourly_wage = st.sidebar.number_input(
        "Operator Wage ($/hr)",
        value=TEA_DEFAULTS_NTH['operator_hourly_wage'], step=0.5,
        help="Hourly wage for operators of the facility ($/hr)"
    )

# TEA Parameters — Default / Custom toggle
# Ultrasound-assisted extraction is a first-of-a-kind (FOAK) process, so the
# mature "Nth Plant" cost assumption doesn't apply — drop it from the options
# whenever the ultrasound reactor is selected.
_ultrasound_foak = (reactor_type == 'ultrasound')
if _ultrasound_foak:
    _tea_mode_options = ["FOAK", "Custom"]
    _tea_mode_help = (
        "Ultrasound is a first-of-a-kind (FOAK) technology, so the mature "
        "'Nth Plant' assumption is unavailable. FOAK: first-of-a-kind "
        "(higher risk/cost). Custom: adjust every value."
    )
else:
    _tea_mode_options = ["Nth Plant", "FOAK", "Custom"]
    _tea_mode_help = (
        "Nth Plant: mature technology defaults. FOAK: first-of-a-kind "
        "(higher risk/cost). Custom: adjust every value."
    )

tea_mode = st.sidebar.radio(
    "TEA Parameters",
    options=_tea_mode_options,
    horizontal=True,
    help=_tea_mode_help,
)

if _ultrasound_foak:
    st.sidebar.caption(
        "ℹ️ Ultrasound is FOAK — 'Nth Plant' is disabled; defaulting to FOAK."
    )

# Select the active defaults dict for Nth Plant / FOAK
if tea_mode == "Nth Plant":
    _d = TEA_DEFAULTS_NTH
elif tea_mode == "FOAK":
    _d = TEA_DEFAULTS_FOAK
else:
    _d = TEA_DEFAULTS_NTH  # used only as initial values for Custom widgets

if tea_mode in ("Nth Plant", "FOAK"):
    IRR = _d['IRR']
    duration = _d['duration']
    operating_days = _d['operating_days']
    supervision_factor = _d['supervision_factor']
    income_tax = _d['income_tax']
    depreciation = _d['depreciation']
    construction_schedule = _d['construction_schedule']
    startup_months = _d['startup_months']
    startup_FOCfrac = _d['startup_FOCfrac']
    startup_VOCfrac = _d['startup_VOCfrac']
    startup_salesfrac = _d['startup_salesfrac']
    maintenance = _d['maintenance']
    property_insurance = _d['property_insurance']
    property_tax = _d['property_tax']
    fringe_benefits = _d['fringe_benefits']
    supplies = _d['supplies']
    administration = _d['administration']
    Additional_direct_costs = _d['Additional_direct_costs']
    indirect_costs = _d['indirect_costs']
    WC_over_FCI = _d['WC_over_FCI']
    finance_interest = _d['finance_interest']
    finance_years = _d['finance_years']
    finance_fraction = _d['finance_fraction']

    with st.sidebar.expander(f"View {tea_mode} Default Values"):
        st.write(f"**IRR:** {IRR}")
        st.write(f"**Plant Life:** {duration} years")
        st.write(f"**Operating Days:** {operating_days}")
        st.write(f"**Depreciation:** {depreciation}")
        st.write(f"**Income Tax:** {income_tax}")
        st.write(f"**Construction Schedule:** {construction_schedule}")
        st.write(f"**Start-up Months:** {startup_months}")
        st.write(f"**Start-up FOC Fraction:** {startup_FOCfrac}")
        st.write(f"**Start-up VOC Fraction:** {startup_VOCfrac}")
        st.write(f"**Start-up Sales Fraction:** {startup_salesfrac}")
        st.write(f"**Maintenance:** {maintenance}")
        st.write(f"**Property Insurance:** {property_insurance}")
        st.write(f"**Property Tax:** {property_tax}")
        st.write(f"**Fringe Benefits:** {fringe_benefits}")
        st.write(f"**Supplies:** {supplies}")
        st.write(f"**Administration:** {administration}")
        st.write(f"**Additional Direct Costs:** {Additional_direct_costs}")
        st.write(f"**Indirect Costs:** {indirect_costs}")
        st.write(f"**Working Capital/FCI:** {WC_over_FCI}")
        st.write(f"**Finance Interest:** {finance_interest}")
        st.write(f"**Finance Years:** {finance_years}")
        st.write(f"**Finance Fraction:** {finance_fraction}")

else:
    # Custom mode — all parameters editable
    with st.sidebar.expander("TEA Parameters", expanded=True):
        IRR = st.number_input(
            "IRR",
            min_value=0.0, max_value=1.0, value=_d['IRR'], format="%.2f",
            help="Internal Rate of Return - the minimum acceptable return on investment."
        )
        duration = st.number_input(
            "Plant Life (years)",
            min_value=1, max_value=50, value=_d['duration'],
            help="Economic lifetime of the facility."
        )
        operating_days = st.number_input(
            "Operating Days/Year",
            min_value=1, max_value=365, value=_d['operating_days'],
            help="Days per year the plant operates."
        )
        supervision_factor = st.number_input(
            "Supervision Factor (fraction of operating labour)",
            value=_d['supervision_factor'], step=0.05,
            help="Cost of supervision labour as fraction of operating labour"
        )
        income_tax = st.number_input(
            "Income Tax Rate",
            min_value=0.0, max_value=1.0, value=_d['income_tax'], format="%.2f",
            help="Corporate income tax rate."
        )
        depreciation = st.selectbox(
            "Depreciation Method",
            options=depreciation_options,
            index=depreciation_options.index(_d['depreciation']),
            help="Tax depreciation schedule. MACRS is standard in the US."
        )
        construction_schedule_str = st.text_input(
            "Construction Schedule (comma-separated fractions)",
            value=", ".join(str(x) for x in _d['construction_schedule']),
            help="Fraction of capital spent each year during construction. Must sum to 1.0."
        )
        # Parse construction schedule string to tuple
        try:
            construction_schedule = tuple(float(x.strip()) for x in construction_schedule_str.split(","))
        except ValueError:
            construction_schedule = _d['construction_schedule']
            st.warning("Invalid construction schedule format. Using default.")

        startup_months = st.number_input(
            "Start-up Months",
            min_value=1, max_value=60, value=_d['startup_months'],
            help="Duration of start-up period before full production."
        )
        startup_FOCfrac = st.number_input(
            "Start-up FOC Fraction",
            min_value=0.0, max_value=2.0, value=_d['startup_FOCfrac'], format="%.2f",
            help="Fixed operating costs during start-up as fraction of normal FOC."
        )
        startup_VOCfrac = st.number_input(
            "Start-up VOC Fraction",
            min_value=0.0, max_value=2.0, value=_d['startup_VOCfrac'], format="%.2f",
            help="Variable operating costs during start-up as fraction of normal VOC."
        )
        startup_salesfrac = st.number_input(
            "Start-up Sales Fraction",
            min_value=0.0, max_value=1.0, value=_d['startup_salesfrac'], format="%.2f",
            help="Revenue during start-up as fraction of normal sales."
        )
        maintenance = st.number_input(
            "Maintenance (fraction of FCI)",
            value=_d['maintenance'], format="%.3f",
            help="Annual maintenance and repair costs as a fraction of Fixed Capital Investment."
        )
        property_insurance = st.number_input(
            "Property Insurance (fraction of FCI)",
            value=_d['property_insurance'], format="%.3f",
            help="Annual insurance premium as a fraction of FCI."
        )
        property_tax = st.number_input(
            "Property Tax (fraction of FCI)",
            value=_d['property_tax'], format="%.3f",
            help="Annual local property taxes as a fraction of FCI."
        )
        fringe_benefits = st.number_input(
            "Fringe Benefits (fraction of labor)",
            value=_d['fringe_benefits'], format="%.2f",
            help="Employee benefits as a fraction of base labor cost."
        )
        supplies = st.number_input(
            "Supplies (fraction of labor)",
            value=_d['supplies'], format="%.2f",
            help="Consumables, PPE, etc. as a fraction of labor cost."
        )
        administration = st.number_input(
            "Administration (fraction of labor)",
            value=_d['administration'], format="%.2f",
            help="Administrative and overhead costs as a fraction of labor cost."
        )
        Additional_direct_costs = st.number_input(
            "Additional Direct Costs (fraction)",
            value=_d['Additional_direct_costs'], format="%.3f",
            help="Site development, buildings, and auxiliary facilities as a fraction of installed equipment cost."
        )
        indirect_costs = st.number_input(
            "Indirect Costs (fraction of DPI)",
            value=_d['indirect_costs'], format="%.2f",
            help="Engineering, construction management, contractor fees, and contingency as a fraction of DPI."
        )
        WC_over_FCI = st.number_input(
            "Working Capital/FCI",
            value=_d['WC_over_FCI'], format="%.3f",
            help="Working capital needed to operate, as a fraction of FCI."
        )
        finance_interest = st.number_input(
            "Finance Interest Rate",
            value=_d['finance_interest'], format="%.2f",
            help="Interest rate on borrowed capital."
        )
        finance_years = st.number_input(
            "Finance Years",
            value=_d['finance_years'],
            help="Number of years to repay the loan."
        )
        finance_fraction = st.number_input(
            "Finance Fraction",
            value=_d['finance_fraction'], format="%.2f",
            help="Fraction of FCI financed through debt (vs. equity)."
        )

# =========================================================================
# Parameter Sweep Configuration (only in Parameter Sweep mode)
# =========================================================================
# Schema of sweepable numeric parameters. Each entry lists the human
# label, hard bounds matching the original widget, a "good" default for
# the sweep range, a printf-style format, and the cast for sweep values
# (None = float, int = round-to-int + dedupe).
NUMERIC_SWEEP_SPECS = {
    # Feed drying (pre-storage drum dryer)
    # Upper bound tracks the selected feedstock: drying can only remove the
    # water that arrives with the feed.
    'feed_dry_moisture':  {'label': 'Feed moisture after drying (frac)', 'min': 0.005, 'max': float(_as_received_moisture), 'def_lo': min(0.04, float(_as_received_moisture)), 'def_hi': min(0.15, float(_as_received_moisture)), 'fmt': '%.4f', 'cast': None, 'group': 'Feed drying'},
    'feed_dry_T':         {'label': 'Feed drying temperature (°C)',      'min': 40.0,  'max': 200.0,'def_lo': 60.0, 'def_hi': 120.0,'fmt': '%.1f',  'cast': None, 'group': 'Feed drying'},
    # Process
    'feedflow':        {'label': 'Feed flow rate (kg/hr)',           'min': 10.0,   'max': 1000000.0, 'def_lo': 200.0,    'def_hi': 800.0,    'fmt': '%.2f',  'cast': None, 'group': 'Process'},
    'solvflow':        {'label': 'Solvent flow rate (kg/hr)',        'min': 10.0,   'max': 1000000.0, 'def_lo': 500.0,    'def_hi': 2000.0,   'fmt': '%.2f',  'cast': None, 'group': 'Process'},
    'ExtractT':        {'label': 'Extraction temperature (°C)',       'min': 0.0,    'max': 100.0,     'def_lo': 25.0,     'def_hi': 65.0,     'fmt': '%.2f',  'cast': None, 'group': 'Process'},
    'extractt':        {'label': 'Extraction time (hr)',              'min': 0.1,    'max': 24.0,      'def_lo': 0.5,      'def_hi': 4.0,      'fmt': '%.3f',  'cast': None, 'group': 'Process'},
    'Particlesize':    {'label': 'Particle radius (cm)',              'min': 0.01,   'max': 5.0,       'def_lo': 0.1,      'def_hi': 1.0,      'fmt': '%.4f',  'cast': None, 'group': 'Process'},
    'ExtractP_custom': {'label': 'Extraction pressure (Pa, Custom mode)', 'min': 1000.0,'max': 1000000.0,'def_lo': 50000.0,'def_hi': 300000.0,'fmt': '%.0f', 'cast': None, 'group': 'Process'},
    # Concentration / drying
    'evap_n_effects':     {'label': 'Evaporator effects',             'min': 1,      'max': 5,         'def_lo': 1,        'def_hi': 5,        'fmt': '%d',    'cast': int,  'group': 'Concentration'},
    'evap_target_solids': {'label': 'Evaporator target solids (frac)','min': 0.10,   'max': 0.50,      'def_lo': 0.20,     'def_hi': 0.45,     'fmt': '%.3f',  'cast': None, 'group': 'Concentration'},
    'dryer_moisture':     {'label': 'Dryer residual solvent (frac)',  'min': 0.001,  'max': 0.20,      'def_lo': 0.02,     'def_hi': 0.10,     'fmt': '%.4f',  'cast': None, 'group': 'Concentration'},
    # Recycle
    'recycle_x_max':      {'label': 'Max impurity frac at inlet',      'min': 0.001,  'max': 0.20,      'def_lo': 0.005,    'def_hi': 0.05,     'fmt': '%.4f',  'cast': None, 'group': 'Recycle'},
    # Economics
    'elec_price':         {'label': 'Electricity price ($/kWh)',       'min': 0.0,    'max': 1.0,       'def_lo': 0.10,     'def_hi': 0.30,     'fmt': '%.4f',  'cast': None, 'group': 'Economics'},
    'feed_price':         {'label': 'Feed price ($/kg)',                'min': 0.0,    'max': 100.0,     'def_lo': 0.0,      'def_hi': 0.50,     'fmt': '%.4f',  'cast': None, 'group': 'Economics'},
    'operator_hourly_wage': {'label': 'Operator wage ($/hr)',           'min': 0.0,    'max': 200.0,     'def_lo': 18.0,     'def_hi': 35.0,     'fmt': '%.2f',  'cast': None, 'group': 'Economics'},
    'feed_storage_days':  {'label': 'Feed storage (days)',              'min': 1,      'max': 30,        'def_lo': 3,        'def_hi': 14,       'fmt': '%d',    'cast': int,  'group': 'Economics'},
    'solv_storage_days':  {'label': 'Solvent storage (days)',           'min': 1,      'max': 30,        'def_lo': 3,        'def_hi': 14,       'fmt': '%d',    'cast': int,  'group': 'Economics'},
    # TEA
    'IRR':                {'label': 'IRR',                              'min': 0.0,    'max': 1.0,       'def_lo': 0.05,     'def_hi': 0.25,     'fmt': '%.3f',  'cast': None, 'group': 'TEA'},
    'duration':           {'label': 'Plant life (yr)',                  'min': 1,      'max': 50,        'def_lo': 15,       'def_hi': 30,       'fmt': '%d',    'cast': int,  'group': 'TEA'},
    'operating_days':     {'label': 'Operating days / yr',              'min': 1,      'max': 365,       'def_lo': 280,      'def_hi': 340,      'fmt': '%d',    'cast': int,  'group': 'TEA'},
    'income_tax':         {'label': 'Income tax rate',                  'min': 0.0,    'max': 1.0,       'def_lo': 0.15,     'def_hi': 0.30,     'fmt': '%.3f',  'cast': None, 'group': 'TEA'},
    'supervision_factor': {'label': 'Supervision factor',               'min': 0.0,    'max': 1.0,       'def_lo': 0.10,     'def_hi': 0.25,     'fmt': '%.3f',  'cast': None, 'group': 'TEA'},
    'maintenance':        {'label': 'Maintenance (frac of FCI)',        'min': 0.0,    'max': 0.20,      'def_lo': 0.02,     'def_hi': 0.05,     'fmt': '%.4f',  'cast': None, 'group': 'TEA'},
    'property_insurance': {'label': 'Property insurance (frac of FCI)',  'min': 0.0,    'max': 0.10,      'def_lo': 0.005,    'def_hi': 0.015,    'fmt': '%.4f',  'cast': None, 'group': 'TEA'},
    'property_tax':       {'label': 'Property tax (frac of FCI)',       'min': 0.0,    'max': 0.10,      'def_lo': 0.005,    'def_hi': 0.020,    'fmt': '%.4f',  'cast': None, 'group': 'TEA'},
    'fringe_benefits':    {'label': 'Fringe benefits (frac of labor)',  'min': 0.0,    'max': 2.0,       'def_lo': 0.20,     'def_hi': 0.60,     'fmt': '%.3f',  'cast': None, 'group': 'TEA'},
    'supplies':           {'label': 'Supplies (frac of labor)',         'min': 0.0,    'max': 2.0,       'def_lo': 0.05,     'def_hi': 0.20,     'fmt': '%.3f',  'cast': None, 'group': 'TEA'},
    'administration':     {'label': 'Administration (frac of labor)',   'min': 0.0,    'max': 2.0,       'def_lo': 0.10,     'def_hi': 0.30,     'fmt': '%.3f',  'cast': None, 'group': 'TEA'},
    'Additional_direct_costs': {'label': "Add'l direct costs (frac)",    'min': 0.0,    'max': 1.0,       'def_lo': 0.10,     'def_hi': 0.25,     'fmt': '%.4f',  'cast': None, 'group': 'TEA'},
    'indirect_costs':     {'label': 'Indirect costs (frac of DPI)',     'min': 0.0,    'max': 2.0,       'def_lo': 0.40,     'def_hi': 0.80,     'fmt': '%.3f',  'cast': None, 'group': 'TEA'},
    'WC_over_FCI':        {'label': 'Working capital / FCI',            'min': 0.0,    'max': 1.0,       'def_lo': 0.03,     'def_hi': 0.10,     'fmt': '%.4f',  'cast': None, 'group': 'TEA'},
    'finance_interest':   {'label': 'Finance interest',                  'min': 0.0,    'max': 1.0,       'def_lo': 0.05,     'def_hi': 0.12,     'fmt': '%.4f',  'cast': None, 'group': 'TEA'},
    'finance_years':      {'label': 'Finance years',                     'min': 1,      'max': 30,        'def_lo': 5,        'def_hi': 15,       'fmt': '%d',    'cast': int,  'group': 'TEA'},
    'finance_fraction':   {'label': 'Finance fraction',                  'min': 0.0,    'max': 1.0,       'def_lo': 0.30,     'def_hi': 0.60,     'fmt': '%.3f',  'cast': None, 'group': 'TEA'},
    'startup_months':     {'label': 'Start-up months',                   'min': 1,      'max': 60,        'def_lo': 3,        'def_hi': 18,       'fmt': '%d',    'cast': int,  'group': 'TEA'},
    'startup_FOCfrac':    {'label': 'Start-up FOC fraction',             'min': 0.0,    'max': 2.0,       'def_lo': 0.75,     'def_hi': 1.25,     'fmt': '%.3f',  'cast': None, 'group': 'TEA'},
    'startup_VOCfrac':    {'label': 'Start-up VOC fraction',             'min': 0.0,    'max': 2.0,       'def_lo': 0.50,     'def_hi': 1.00,     'fmt': '%.3f',  'cast': None, 'group': 'TEA'},
    'startup_salesfrac':  {'label': 'Start-up sales fraction',           'min': 0.0,    'max': 1.0,       'def_lo': 0.30,     'def_hi': 0.70,     'fmt': '%.3f',  'cast': None, 'group': 'TEA'},
}

CATEGORICAL_SWEEP_SPECS = {
    'selected_feedstock': {'label': 'Feedstock',     'options': list(feedstocks.keys())},
    'solv':               {'label': 'Solvent',       'options': list(solvprice_index.keys())},
    'reactor_type':       {'label': 'Reactor type',  'options': ['conventional', 'ultrasound', 'microwave']},
    'heatutility':        {'label': 'Heat utility',  'options': HeatU},
    'depreciation':       {'label': 'Depreciation',  'options': depreciation_options},
}

# Build sweep specs from sidebar widgets (only consulted in sweep mode)
sweep_values = {}       # {param_name: [v1, v2, ...]}
sweep_summary_labels = {}

if sim_mode == "Parameter Sweep":
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔄 Parameter Sweep")
    st.sidebar.caption(
        "Tick the parameters to vary. Each unticked parameter holds the "
        "single value set above. Ranges below override those for swept "
        "parameters only."
    )

    # Numeric sweeps, organised by group
    groups_ordered = ['Process', 'Feed drying', 'Concentration', 'Recycle',
                      'Economics', 'TEA']
    for group in groups_ordered:
        group_keys = [k for k, s in NUMERIC_SWEEP_SPECS.items()
                      if s['group'] == group]
        if not group_keys:
            continue
        # Count active so header can show a quick badge
        n_active = sum(1 for k in group_keys
                       if st.session_state.get(f'swp_{k}', False))
        header = f"{group}  ·  {n_active} swept" if n_active else group
        with st.sidebar.expander(header, expanded=False):
            for k in group_keys:
                spec = NUMERIC_SWEEP_SPECS[k]
                enabled = st.checkbox(
                    f"Sweep **{spec['label']}**",
                    key=f"swp_{k}",
                )
                if enabled:
                    c1, c2 = st.columns(2)
                    with c1:
                        smin = st.number_input(
                            "Min",
                            min_value=float(spec['min']),
                            max_value=float(spec['max']),
                            value=float(spec['def_lo']),
                            format=spec['fmt'],
                            key=f"swp_{k}_min",
                        )
                        n_pts = st.number_input(
                            "# points",
                            min_value=2, max_value=100, value=5, step=1,
                            key=f"swp_{k}_n",
                        )
                    with c2:
                        smax = st.number_input(
                            "Max",
                            min_value=float(spec['min']),
                            max_value=float(spec['max']),
                            value=float(spec['def_hi']),
                            format=spec['fmt'],
                            key=f"swp_{k}_max",
                        )
                        scale = st.selectbox(
                            "Scale",
                            options=['linear', 'log'],
                            key=f"swp_{k}_scale",
                            help=("Linear: evenly spaced. Log: "
                                  "log-spaced (both endpoints must be "
                                  "positive — falls back to linear "
                                  "otherwise)."),
                        )
                    if smax <= smin:
                        st.warning(
                            f"Max must be greater than Min for {spec['label']}."
                        )
                    else:
                        vals = _make_sweep_values({
                            'min': smin, 'max': smax, 'n': n_pts,
                            'scale': scale, 'cast': spec['cast'],
                        })
                        sweep_values[k] = vals
                        sweep_summary_labels[k] = spec['label']
                        # Compact preview
                        preview = ", ".join(
                            (spec['fmt'] % v) if spec['cast'] is not int
                            else str(v)
                            for v in vals[:6]
                        )
                        if len(vals) > 6:
                            preview += ", ..."
                        st.caption(f"→ {len(vals)} value(s): {preview}")

    # Categorical sweeps
    with st.sidebar.expander("Categorical sweeps", expanded=False):
        for k, spec in CATEGORICAL_SWEEP_SPECS.items():
            enabled = st.checkbox(
                f"Sweep **{spec['label']}**",
                key=f"swp_{k}",
            )
            if enabled:
                chosen = st.multiselect(
                    spec['label'],
                    options=spec['options'],
                    key=f"swp_{k}_vals",
                )
                if chosen:
                    sweep_values[k] = list(chosen)
                    sweep_summary_labels[k] = spec['label']
                    st.caption(f"→ {len(chosen)} value(s)")
                else:
                    st.caption("Pick at least one value to include.")

    # Top-level summary
    if sweep_values:
        n_total = 1
        for vals in sweep_values.values():
            n_total *= len(vals)
        st.sidebar.success(
            f"📊 **{n_total} run(s)** across "
            f"{len(sweep_values)} swept parameter(s)"
        )
        if n_total > 200:
            st.sidebar.warning(
                f"⚠️ {n_total} runs may take a long time — consider "
                "reducing the grid."
            )
    else:
        st.sidebar.info("Tick at least one parameter to enable the sweep.")

# Run button — label depends on mode
if sim_mode == "Single Run":
    run_simulation = st.sidebar.button(
        "🚀 Run Simulation", type="primary", use_container_width=True
    )
    run_sweep = False
else:
    run_sweep = st.sidebar.button(
        "🚀 Run Sweep",
        type="primary",
        use_container_width=True,
        disabled=not sweep_values,
    )
    run_simulation = False

# =========================================================================
# Feedstock Composition Display (main area)
# =========================================================================
with st.expander("🌿 Feedstock Composition — " + selected_feedstock, expanded=False):
    fs_info = feedstocks[selected_feedstock]
    col_a, col_b = st.columns([1, 2])
    with col_a:
        st.write(f"**Feedstock Type:** {fs_info['type']}")
        st.write(f"**Moisture Content:** {fs_info['moisture'] * 100:.1f}%")
        st.write(f"**Number of Components:** {len(fs_info['chems'])}")
    with col_b:
        st.write(f"**Dry-basis composition shown in mg/g**")

    # Build composition table with name next to value (mg/g)
    comp_table = []
    for entry, comp_val in zip(fs_info['chems'], fs_info['comp']):
        chem_name = _get_chem_name(entry)
        mg_per_g = comp_val * 1000  # convert mass fraction to mg/g
        comp_table.append({
            'Chemical': chem_name,
            'Composition (mg/g)': _fmt_mg_g(mg_per_g),
        })
    df_feedstock = pd.DataFrame(comp_table)
    st.dataframe(df_feedstock, use_container_width=True, hide_index=True, height=500)

# Main content area
def _run_one_simulation(p, *, display=True):
    """Run one full simulation with the given parameters.

    Parameters
    ----------
    p : dict
        Inputs keyed by the original script's variable names
        (feedflow, ExtractT, IRR, ...).
    display : bool
        True (single-run mode): render the full Streamlit report —
        tabs, metrics, plots, stream tables.
        False (sweep mode): suppress all per-run Streamlit output and
        only return the scalar metrics. Errors are raised so the outer
        sweep loop can record them.

    Returns
    -------
    scalars : dict | None
        Scalar metrics suitable for one CSV row. None only when
        display=True and an error was caught (and shown in the UI).
    """
    # ── Unpack inputs ──────────────────────────────────────────────────
    selected_feedstock      = p['selected_feedstock']
    feedflow                = p['feedflow']
    solv                    = p['solv']
    solvflow                = p['solvflow']
    # In solvent/feed-ratio mode, derive the solvent flow from the (possibly
    # swept) feed flow so the ratio is held constant. Falls back gracefully
    # for params dicts that predate this option.
    if p.get('solv_flow_mode') == "Solvent/feed ratio":
        _ratio = p.get('solv_feed_ratio')
        if _ratio is not None:
            solvflow = _ratio * feedflow
    ExtractT                = p['ExtractT']
    extractt                = p['extractt']
    Particlesize            = p['Particlesize']
    reactor_type            = p['reactor_type']
    heatutility             = p['heatutility']
    pressure_mode           = p['pressure_mode']
    ExtractP_custom         = p.get('ExtractP_custom')
    feed_dry_enable         = p.get('feed_dry_enable', True)
    feed_dry_moisture       = p.get('feed_dry_moisture', 0.06)
    feed_dry_T              = p.get('feed_dry_T', 90.0)
    evap_n_effects          = int(p['evap_n_effects'])
    evap_target_solids      = p['evap_target_solids']
    dryer_moisture          = p['dryer_moisture']
    enable_recycle          = p['enable_recycle']
    recycle_x_max           = p['recycle_x_max']
    Yearstart               = p['Yearstart']
    elec_price              = p['elec_price']
    feed_price              = p['feed_price']
    feed_storage_days       = int(p['feed_storage_days'])
    solv_storage_days       = int(p['solv_storage_days'])
    operator_hourly_wage    = p['operator_hourly_wage']
    IRR                     = p['IRR']
    duration                = int(p['duration'])
    operating_days          = int(p['operating_days'])
    supervision_factor      = p['supervision_factor']
    income_tax              = p['income_tax']
    depreciation            = p['depreciation']
    construction_schedule   = p['construction_schedule']
    startup_months          = int(p['startup_months'])
    startup_FOCfrac         = p['startup_FOCfrac']
    startup_VOCfrac         = p['startup_VOCfrac']
    startup_salesfrac       = p['startup_salesfrac']
    maintenance             = p['maintenance']
    property_insurance      = p['property_insurance']
    property_tax            = p['property_tax']
    fringe_benefits         = p['fringe_benefits']
    supplies                = p['supplies']
    administration          = p['administration']
    Additional_direct_costs = p['Additional_direct_costs']
    indirect_costs          = p['indirect_costs']
    WC_over_FCI             = p['WC_over_FCI']
    finance_interest        = p['finance_interest']
    finance_years           = int(p['finance_years'])
    finance_fraction        = p['finance_fraction']

    _spinner_ctx = (st.spinner("Running simulation... This may take a moment.")
                    if display else contextlib.nullcontext())
    with _spinner_ctx:
        try:
            # ── Progress bar ─────────────────────────────────────────────
            # The run proceeds as a sequence of stages; the bar and the
            # status line below it show which stage is active and the
            # overall % complete (by stage count).
            _progress_stages = [
                "Setting up chemicals and thermodynamics",
                "Building flowsheet",
                "Priming recycle loop",
                "Solving recycle (split for impurity cap)",
                "Running techno-economic analysis",
                "Preparing results",
            ]
            _progress_bar = st.progress(0.0) if display else _NoopUI()
            _progress_status = st.empty() if display else _NoopUI()
            _progress_state = {'i': 0}
            # Final one-line status messages (recycle solved, evaporator
            # solved, simulation complete) are collected here during the run
            # and rendered together BELOW the results, so they don't stack at
            # the top and push the results down.
            _run_status_lines = []

            def _advance_progress(label=None):
                i = _progress_state['i']
                n = len(_progress_stages)
                if i < n:
                    name = label or _progress_stages[i]
                    _progress_bar.progress(i / n)
                    _progress_status.info(
                        f"Step {i + 1}/{n}: {name} ...")
                    _progress_state['i'] += 1

            def _finish_progress():
                # Clear the live progress bar and status line entirely rather
                # than leaving a full bar + "All steps complete" lingering at
                # the top of the page. The run-summary block at the bottom is
                # the finished-state confirmation instead.
                _progress_bar.empty()
                _progress_status.empty()

            _advance_progress()  # Stage 1: chemicals & thermo

            # Get feedstock data
            feedchem_entries = feedstocks[selected_feedstock]['chems']  # list of dicts (name/cas/chebi)
            feedchems = [_get_chem_name(c) for c in feedchem_entries]   # plain name list
            feedcomp = feedstocks[selected_feedstock]['comp']
            moisture = feedstocks[selected_feedstock]['moisture']
            # The drum dryer removes water; it cannot add it. If a sweep
            # pairs a drying target with a feedstock that arrives drier
            # than that target, hold the target at the incoming moisture
            # (the dryer then passes the feed straight through).
            feed_dry_moisture = min(feed_dry_moisture, moisture)
            feedstock_type = feedstocks[selected_feedstock]['type']
            porosity_tortuosity_ratio = porosity_torosity[feedstock_type]

            # Get solvent price
            solv_price = solvprice_index[solv]

            # Define chemicals
            Solvchems = ['ethanol', 'hexane', 'acetone', 'water', 'chloroform']

            # ── Build combined chemical list for the proxy finder ────────────────
            # Feedstock compounds keep their rich dict format (name/cas/chebi)
            # so the proxy finder can use all three identifiers for lookup.
            # Solvents and water are plain strings (always native in thermosteam).
            seen_names = set()
            all_chemical_entries = []  # mixed list: dicts for feedstock chems, strings for solvents
            for entry in feedchem_entries:
                name = _get_chem_name(entry)
                if name not in seen_names:
                    all_chemical_entries.append(entry)
                    seen_names.add(name)
            for name in Solvchems + ['water']:
                if name not in seen_names:
                    all_chemical_entries.append(name)
                    seen_names.add(name)

            # ── Use proxy finder to resolve chemicals ──────────────────────────
            proxy_chemicals, proxy_report, name_map = assign_proxies(all_chemical_entries)

            # ── Build reverse mapping: original_user_name → thermosteam_id ────
            orig_to_ts = {orig: ts_id for ts_id, orig in name_map.items()}

            # ── Flag any proxy compounds ──────────────────────────────────────
            proxy_warnings = []
            # Volatility-screen display data, populated when the screen runs
            # (below) and rendered at the bottom of the output. Initialized
            # here so the bottom render is safe on every code path.
            _volatility_screen_info = None
            for name, match in proxy_report.items():
                if not match.in_thermosteam and match.proxy_name:
                    proxy_warnings.append(f"'{name}' not in database — proxy '{match.proxy_name}' used")
                elif not match.in_thermosteam and match.proxy_name is None:
                    proxy_warnings.append(f"'{name}' not in database — no proxy found, compound missing")

            # ── Name translation helpers ──────────────────────────────────────
            def ts_to_original(ts_id):
                """Translate a thermosteam chemical ID back to the user's original name."""
                return name_map.get(ts_id, ts_id)

            def original_to_ts_func(orig_name):
                """Translate a user's original chemical name to its thermosteam ID."""
                return orig_to_ts.get(orig_name, orig_name)

            # Set up thermodynamics using proxy-resolved chemicals
            chemicals_obj = proxy_chemicals

            # ── Volatility screen: which species may enter VLE ───────────────
            # The VLE must KEEP genuinely volatile solutes. They co-evaporate
            # with the solvent, land in the condensate, and ARE the "impurity"
            # that recycle_x_max caps. Locking every solute to the liquid phase
            # would make the condensate pure solvent, send f_solv -> 1, and
            # cause the impurity-cap bound in adjust_recycle_split to become
            # infinite and never bind -- silently disabling the purge spec.
            #
            # But species that are non-volatile IN PRACTICE must be excluded,
            # or the flash tries to boil them. As V -> 1 the mixture dew point
            # is set by the HEAVIEST species present, so the solver marches T
            # toward cinnamic acid's boiling point (~300 C) and past it, then
            # dies extrapolating Hvap:
            #   RuntimeError: Failed to extrapolate enthalpy of vaporization
            #   method 'MORGAN_KOBAYASHI' at T=906.83 K for CASRN '140-10-3'
            # That is the exception CorrectedMEE's GUARD 1 has been swallowing.
            #
            # The OLD screen (Psat < 1e-5 Pa at a fixed 100 C) used the wrong
            # QUANTITY, not merely the wrong number:
            #   * Absolute Psat says nothing about whether a species competes
            #     with the SOLVENT for the vapour. A few Pa is negligible
            #     against ethanol (~100 kPa at Tb) but significant against a
            #     heavy solvent -- so any absolute cutoff tuned for one solvent
            #     is wrong for the next one someone selects.
            #   * 373.15 K is not the evaporator temperature.
            # Cinnamic acid (Psat(100 C) ~ Pa, i.e. >> 1e-5) therefore CLEARED
            # the screen, stayed in the VLE, and wrecked the flash.
            #
            # CORRECT SCREEN: relative volatility against the ACTUAL solvent at
            # the ACTUAL evaporator temperature.
            #       alpha_i = Psat_i(T_evap) / Psat_solvent(T_evap)
            # Keep alpha >= ALPHA_MIN (can reach the condensate, must be
            # tracked); lock the rest to liquid. This rescales itself for any
            # solvent and any operating temperature, so it generalizes.
            #
            # ALPHA_MIN = 1e-4 is tied to what the model actually needs to
            # resolve: a solute at alpha = 1e-4 and liquid mole fraction 0.01
            # contributes ~1 ppm to the vapour -- three orders of magnitude
            # below a 0.5% (5,000 ppm) impurity cap, so it cannot move the
            # recycle balance. Volatile solutes worth keeping (limonene,
            # furfural, acetic acid, light phenolics) sit at alpha ~1e-2 to 1.
            ALPHA_MIN = 1e-4

            solvent_names_lower = {s.lower() for s in Solvchems + ['water']}

            # Screen at the hottest temperature the flowsheet reaches: the
            # evaporator's first effect sits at the solvent boiling point, or
            # at the extractor outlet if that is hotter (cf. suggest_pressures).
            try:
                _solv_chem = tmo.Chemical(solv)
                _solv_Tb = _solv_chem.Tb or 373.15
                T_screen = max(_solv_Tb, ExtractT + 273.15 + 2.0)
                _Psat_solv_screen = _solv_chem.Psat(T_screen)
            except Exception:
                T_screen = 373.15
                _Psat_solv_screen = None

            phase_locked_for_evap = []      # (ID, alpha) excluded from VLE
            kept_volatile = []              # (ID, alpha) retained in VLE

            for i, chem in enumerate(chemicals_obj):
                if isinstance(chem, str):
                    try:
                        chem_obj = tmo.Chemical(chem)
                    except Exception:
                        continue
                else:
                    chem_obj = chem

                # Solvents and water are the volatile species by definition.
                if chem_obj.ID.lower() in solvent_names_lower:
                    continue
                if getattr(chem_obj, 'locked_state', None):
                    continue

                # Relative volatility at evaporator temperature. If Psat cannot
                # even be evaluated there (common for proxied chemicals with
                # extrapolated correlations), treat as non-volatile AND report
                # it -- an unevaluable Psat is exactly what later blows up the
                # flash, so it must not be left in the VLE by default.
                alpha = None
                if _Psat_solv_screen:
                    try:
                        alpha = chem_obj.Psat(T_screen) / _Psat_solv_screen
                    except Exception:
                        alpha = None

                if alpha is not None and alpha >= ALPHA_MIN:
                    kept_volatile.append((chem_obj.ID, alpha))
                    continue    # stays in the VLE: it can reach the condensate

                # Below the cutoff (or unevaluable) -> lock to liquid.
                _a = alpha if alpha is not None else float('nan')
                if isinstance(chemicals_obj[i], str):
                    try:
                        chemicals_obj[i] = tmo.Chemical(chemicals_obj[i],
                                                        phase='l')
                        phase_locked_for_evap.append((chem_obj.ID, _a))
                    except Exception:
                        pass
                else:
                    try:
                        locked = chem_obj.copy(chem_obj.ID, phase='l')
                        chemicals_obj[i] = locked
                        phase_locked_for_evap.append((chem_obj.ID, _a))
                    except Exception:
                        try:
                            locked = tmo.Chemical.blank(chem_obj.ID, phase='l')
                            locked.default()
                            chemicals_obj[i] = locked
                            phase_locked_for_evap.append((chem_obj.ID, _a))
                        except Exception:
                            # Could not lock -> STAYS in the VLE and can still
                            # drive the flash out of range. Do not bury this.
                            warnings.warn(
                                f"Could not phase-lock {chem_obj.ID} to "
                                f"liquid; it remains in the VLE and may drive "
                                f"the evaporator flash out of range.")
            print(f"Native: {sum(1 for m in proxy_report.values() if m.in_thermosteam)}")
            print(f"Proxied: {sum(1 for m in proxy_report.values() if not m.in_thermosteam and m.proxy_name)}")
            for name, m in proxy_report.items():
                if not m.in_thermosteam and m.proxy_name:
                    print(f"  {name} → {m.proxy_name}")
            thermo = tmo.Thermo(
                chemicals=chemicals_obj,
                Gamma=tmo.DortmundActivityCoefficients,
                Phi=tmo.equilibrium.RKFugacityCoefficients,
                PCF=tmo.equilibrium.IdealGasPoyintingCorrectionFactors
            )
            tmo.settings.set_thermo(thermo)
            bst.settings.set_thermo(thermo)

            # Volatility-screen results are RENDERED at the very bottom of the
            # output (next to the proxy-compound warnings), not here — the
            # screen must run now to phase-lock chemicals before the flowsheet
            # is built, but the report reads better grouped with the other
            # chemistry notes at the end. Stash what the display needs.
            _volatility_screen_info = {
                'kept': list(kept_volatile),
                'locked': list(phase_locked_for_evap),
                'solv': solv,
                'T_screen_C': T_screen - 273.15,
                'alpha_min': ALPHA_MIN,
            }

            # Calculate extraction pressure based on mode selection
            if pressure_mode == "Calculated":
                atmospheric_pressure = 101325  # Pa
                solvent_chemical = tmo.settings.chemicals[solv]
                vapor_pressure_at_T = solvent_chemical.Psat(ExtractT + 273.15)
                P_above_boiling = vapor_pressure_at_T * 1.05
                ExtractP = max(atmospheric_pressure, P_above_boiling)
            else:  # Custom pressure
                ExtractP = ExtractP_custom

            # Storage times
            Fstoret = feed_storage_days * 24
            Sstoret = solv_storage_days * 24

            # Set CEPCI and utilities
            bst.settings.CEPCI = cepci_index[Yearstart]
            bst.HeatUtility.default_heating_agents() #reset defaults
            steam_utility = bst.settings.get_agent(heatutility)
            bst.settings.heating_agents = [steam_utility]
            bst.settings.electricity_price = elec_price

            # Set up flowsheet
            _advance_progress()  # Stage 2: building flowsheet
            bst.main_flowsheet.clear()
            bst.main_flowsheet.set_flowsheet('simple_extract')

            # Define biomass group — translate user names to thermosteam IDs
            feedchems_ts = [orig_to_ts.get(name, name) for name in feedchems]
            bst.settings.chemicals.define_group(
                name='Biomass',
                IDs=feedchems_ts,
                composition=feedcomp,
                wt=True
            )

            # Define streams
            feed = bst.Stream(
                'Feed',
                Biomass=1 - moisture,
                Water=moisture,
                total_flow=feedflow,
                units='kg/hr',
                price=feed_price,
                phase='s'
            )

            # Solvent stream — when recycle is enabled this is the FRESH
            # MAKEUP (the only solvent flow that incurs a purchase cost);
            # when recycle is disabled it is the once-through extractor feed.
            # The M1 specification below will adjust the makeup flow at
            # convergence so the combined (recycle + makeup) stream delivers
            # `solvflow` kg/hr of solvent to the extractor.
            solvent = bst.Stream(
                'Solvent',
                **{solv: 1},
                total_flow=solvflow,
                price=solv_price,
                units='kg/hr'
            )

            # Feedstock drying (pre-storage) streams
            feed_dryer_vapor = bst.Stream('feed_dryer_vapor')  # water vented
            dried_feed = bst.Stream('dried_feed')      # out of the drum dryer
            cooled_feed = bst.Stream('cooled_feed')    # cooled → to storage

            solid_residual = bst.Stream('solid_residual')
            extract = bst.Stream('extract')  # intermediate → goes to evaporator
            condensate = bst.Stream('condensate')  # recovered solvent
            dryer_vapor = bst.Stream('dryer_vapor')
            dried_product = bst.Stream('dried_product')  # final product
            cooled_product = bst.Stream('cooled_product')  # cooled dried product

            # ── Recycle-loop streams (used only when enable_recycle=True) ──
            # `recycle` is a tear stream — its initial flow is just an
            # initial guess for the iterative solver. The Wegstein method
            # converges the loop, then the outer brentq loop sizes the
            # purge to satisfy the impurity cap.
            if enable_recycle:
                recycle = bst.Stream('recycle',
                                     **{solv: 0.2 * solvflow,
                                        'Water': 0.01 * solvflow},
                                     units='kg/hr')
                purge = bst.Stream('purge')
                condensed_dryer_vapor = bst.Stream('condensed_dryer_vapor')
                mixed_recycle = bst.Stream('mixed_recycle')

            # Define units
            # ── Feedstock drying (upstream of storage) ────────────────
            # The as-received feed is dried in a rotary drum dryer and
            # cooled back to ambient BEFORE it enters the storage tank, so
            # that (a) the stored pile is stable (no mould / self-heating /
            # loss of extractables), and (b) the feed moisture never enters
            # the extractor, where it would dilute the extract, load the
            # evaporator, force a bigger purge to hold the recycle impurity
            # cap, and end up as water in the dried product.
            #
            #   feed → D101 (drum dryer) → C101 (cooler) → S101 → U101 → U102
            #             └→ feed_dryer_vapor (vented)
            #
            # Drying is skipped entirely when disabled, in which case the
            # wet feed goes straight to storage (original topology).
            if feed_dry_enable:
                D101 = FeedDrumDryer('Feedstock drum dryer',
                    ins=feed,
                    outs=[feed_dryer_vapor, dried_feed],
                    moisture_ID='Water',
                    moisture_content=feed_dry_moisture,
                    T=feed_dry_T + 273.15,
                    thermal_efficiency=feed_dryer_efficiency,
                    U=feed_dryer_U)
                C101 = SolidCooler('Dried feed cooler',
                    ins=dried_feed,
                    outs=[cooled_feed],
                    T_target=FEED_COOL_T,
                    U=solid_cooler_U)
                _feed_to_storage = C101 - 0
            else:
                D101 = None
                C101 = None
                _feed_to_storage = feed

            S101 = units.StorageTank('Feedstock storage', _feed_to_storage,
                                     tau=Fstoret)
            U101 = units.ConveyingBelt('Feed transport', S101 - 0)
            U102 = Mill('Feed grinding', U101 - 0, particle_radius=Particlesize)

            if enable_recycle:
                # Topology: recycle joins the fresh solvent stream
                # AFTER the storage tank — the tank holds purchased
                # fresh makeup only, then the combined stream flows
                # through pump → heater → extractor.
                #
                #     solvent → S102 ──┐
                #                      ├─► M1 → U103 → U105 → E201
                #          recycle ──┘
                S102 = units.StorageTank('Solvent storage', solvent,
                                         tau=Sstoret)
                M1 = bst.Mixer('Solvent Mixer',
                               ins=(recycle, S102 - 0),
                               outs='solvent_in')
                U103 = units.Pump('Solvent Pump', M1 - 0, P=ExtractP)
                U105 = units.HXutility('Solvent Heater', U103 - 0,
                                       T=ExtractT + 273.15)
            else:
                # Open-loop: storage/pump/heater once-through on the
                # fresh solvent feed (original topology).
                S102 = units.StorageTank('Solvent storage', solvent,
                                         tau=Sstoret)
                U103 = units.Pump('Solvent Pump', S102 - 0, P=ExtractP)
                U105 = units.HXutility('Solvent Heater', U103 - 0,
                                       T=ExtractT + 273.15)

            E201 = SolidSolventExtractor('Extractor',
                                         ins=[U102 - 0, U105 - 0],
                                         outs=[solid_residual, extract],
                                         tau=extractt,
                                         solvent_ID=solv,
                                         reactor_type=reactor_type,
                                         particle_radius=Particlesize,
                                         porosity_tortuosity_ratio=porosity_tortuosity_ratio)

            # ── Evaporator + Spray Dryer ──────────────────────────────
            solvent_CAS = tmo.settings.chemicals[solv].CAS
            evap_P = suggest_pressures(solv, n_effects=evap_n_effects,
                                       delta_T_C=20, feed_T_C=ExtractT)

            E301 = CorrectedMEE('Evaporator',
                ins=extract,
                outs=('concentrated', condensate),
                V=0.1, V_definition='Overall',
                P=evap_P, chemical=solvent_CAS, flash=True)

            # Start the diagnostic counters from zero for this simulation.
            E301.reset_counters()

            MAX_EVAP_SOLIDS = 0.50
            _evap_target = min(evap_target_solids, MAX_EVAP_SOLIDS)
            _evap_target_solvent_frac = 1.0 - _evap_target

            # Warm-start cache: the V that solved the spec last time,
            # reused as the centre of a narrow bracket on the next call.
            _last_V = [None]

            # ── Log-space search variable ────────────────────────────
            # The spec is a tolerance on the concentrate's SOLVENT MASS
            # FRACTION, which depends on the solvent LEFT BEHIND -- and
            # that is (1 - V) times the feed, i.e. a difference of two
            # large numbers once V -> 1 (which is exactly where a dilute,
            # high-solvent-flow extract puts it). Bisecting V linearly on
            # [0.1, 0.999] resolves V to only ~2e-4 in 12 steps, while
            # holding the solids fraction to 1e-3 at V ~ 0.99 needs V to
            # ~1e-5. The tolerance is therefore UNREACHABLE: the early
            # exits never fire, every pass burns the full 16 evaluations,
            # and the _last_V warm start never pays off.
            #
            # Searching u = -log10(1 - V) instead gives constant RELATIVE
            # resolution on the leftover solvent -- precisely what the
            # tolerance measures. V=0.9 -> u=1, V=0.99 -> u=2,
            # V=0.999 -> u=3, so 14 steps on the full u-range resolve
            # (1 - V) to ~0.01% and the early exit becomes reachable.
            _V_MIN, _V_MAX = 0.1, 0.999

            def _V_to_u(V):
                V = min(max(V, 0.0), _V_MAX)
                return -log10(max(1e-12, 1.0 - V))

            def _u_to_V(u):
                return 1.0 - 10.0 ** (-u)

            _U_MIN, _U_MAX = _V_to_u(_V_MIN), _V_to_u(_V_MAX)

            # Warm-start cache: the V that solved the spec last time,
            # reused as the centre of a narrow bracket on the next call.
            _last_V = [None]

            @E301.add_specification(run=False)
            def adjust_evap_V():
                """Bisect E301.V (in log space, see above) so the
                concentrated outlet hits the solids target. Speed-ups vs a
                naive wide bisection: an early exit once the achieved
                solvent fraction is within tolerance, and a warm-start
                narrow bracket centred on the last solved V (the
                evaporator feed changes only modestly between recycle
                iterations)."""
                feed_in = E301.ins[0]

                def get_solvent_frac_at_u(u):
                    # Leaves E301 RUN at this u -- so whichever u is
                    # accepted last is the state the unit is left in.
                    # (The old code could return after accepting `lo`
                    # while the unit was still holding the `hi` run.)
                    E301.V = _u_to_V(u)
                    E301._run()
                    conc = E301.outs[0]
                    if conc.F_mass <= 0:
                        return 0.0
                    return conc.imass[solv] / conc.F_mass

                # Initial guess for the warm-start bracket
                if _last_V[0] is not None:
                    V_guess = _last_V[0]
                else:
                    m_feed = feed_in.F_mass
                    if m_feed <= 1e-9:
                        E301.V = 0.5
                        E301._run()
                        return
                    m_solv   = feed_in.imass[solv]
                    m_solids = m_feed - m_solv
                    if m_solids <= 1e-9:
                        V_guess = _V_MAX
                    else:
                        m_conc_target = m_solids / _evap_target
                        m_solv_out    = max(0.0, m_conc_target - m_solids)
                        m_evap        = max(0.0, m_solv - m_solv_out)
                        V_guess = min(_V_MAX, max(1e-3, m_evap / m_feed))

                target = _evap_target_solvent_frac
                _FRAC_TOL = 1e-3

                # solvent_frac falls as V (hence u) rises, so the bracket
                # logic below is unchanged in direction from the linear
                # version -- only the variable has changed.
                u_guess = _V_to_u(V_guess)

                # Narrow bracket: +/-0.3 in u == a factor of ~2 either way
                # in the leftover solvent (1 - V).
                half_width = 0.3
                lo = max(_U_MIN, u_guess - half_width)
                hi = min(_U_MAX, u_guess + half_width)

                # Evaluate lo, CHECK, then hi -- in that order, so an
                # accepted bound is also the run E301 is left holding.
                f_lo = get_solvent_frac_at_u(lo)
                if abs(f_lo - target) < _FRAC_TOL:
                    _last_V[0] = _u_to_V(lo)
                    return
                f_hi = get_solvent_frac_at_u(hi)
                if abs(f_hi - target) < _FRAC_TOL:
                    _last_V[0] = _u_to_V(hi)
                    return

                # Fall back to the wide bracket if the narrow one
                # doesn't bracket-cross
                if (f_lo - target) * (f_hi - target) >= 0:
                    lo, hi = _U_MIN, _U_MAX
                    f_lo = get_solvent_frac_at_u(lo)
                    if abs(f_lo - target) < _FRAC_TOL:
                        _last_V[0] = _u_to_V(lo)
                        return
                    f_hi = get_solvent_frac_at_u(hi)
                    if abs(f_hi - target) < _FRAC_TOL:
                        _last_V[0] = _u_to_V(hi)
                        return
                    if (f_lo - target) * (f_hi - target) >= 0:
                        # Target unreachable anywhere in [V_MIN, V_MAX]:
                        # settle on the closest end and LEAVE E301 RUN
                        # there, so E301.V and E301.outs stay consistent.
                        u_end = (hi if abs(f_hi - target) < abs(f_lo - target)
                                 else lo)
                        get_solvent_frac_at_u(u_end)
                        _last_V[0] = _u_to_V(u_end)
                        return

                # Bisection with early exit
                for _ in range(14):
                    mid = (lo + hi) / 2
                    f_mid = get_solvent_frac_at_u(mid)
                    if abs(f_mid - target) < _FRAC_TOL:
                        _last_V[0] = _u_to_V(mid)
                        return
                    if f_mid > target:      # too much solvent left -> boil more
                        lo = mid
                    else:
                        hi = mid
                # Exhausted: re-run at the final midpoint so that E301.V
                # and E301.outs agree (the old code assigned E301.V without
                # re-running, leaving the outlet from a different V).
                u_final = (lo + hi) / 2
                get_solvent_frac_at_u(u_final)
                _last_V[0] = _u_to_V(u_final)

            SD1 = SolventSprayDryer('Spray Dryer',
                ins=E301.outs[0],
                outs=[dryer_vapor, dried_product],
                solvent=solv,
                moisture_content=dryer_moisture,
                T=None,
                thermal_efficiency=0.218)

            # ── Cooler ────────────────────────────────────────────
            # Product cooler — indirect rotary drum cooler (solid)
            SC1 = SolidCooler('Product cooler',
                ins=dried_product,
                outs=[cooled_product],
                T_target=298.15,  # 25 °C (room temperature)
                U=solid_cooler_U)

            # ── Solvent-recycle units ─────────────────────────────
            # H1 condenses the dryer vapor (gas) to a liquid so it can mix
            # with the evaporator condensate. M2 combines both condensate
            # streams. S1 splits the combined flow between the recycle and
            # the purge — the purge fraction is set by the outer loop to
            # cap the impurity build-up in the loop.
            if enable_recycle:
                H1 = units.HXutility('Dryer vapor condenser',
                                     ins=dryer_vapor,
                                     outs=condensed_dryer_vapor,
                                     V=0.0, rigorous=False)
                M2 = bst.Mixer('Recycle Mixer',
                               ins=(condensate, H1 - 0),
                               outs=mixed_recycle)
                S1 = bst.Splitter('Recycle Splitter',
                                  ins=mixed_recycle,
                                  outs=(recycle, purge),
                                  split=0.5)

                # Inner-loop spec: the fresh makeup tracks the recycle so
                # the combined solvent flow at M1's outlet equals the
                # target (see adjust_makeup below).
                target_solvent_flow = solvflow
                # Small absolute floor on the makeup, so during Wegstein
                # iterations the fresh solvent stream is never empty -
                # SolidSolventExtractor checks the union of feeds for the
                # solvent and raises if both are < 1e-10 mol.
                # Purge floor. This caps S1.split at (1 - _MIN_PURGE_FRAC),
                # i.e. limits recycle to 99.9%. It is a SOLVER guard, not a
                # process requirement: as the split -> 1 the fresh makeup -> 0
                # and the recycle becomes nearly the entire solvent feed, so
                # the loop gain approaches unity and Wegstein can get stiff
                # near the singular point. A small floor keeps a guaranteed
                # exit so the loop never reaches split == 1 exactly.
                #
                # It was previously 0.01 (99% cap) because, when the split was
                # found by a root-search, the search stalled near the upper
                # limit at high flows. That search has since been replaced by
                # the direct mass balance below (adjust_recycle_split solves
                # the split in one shot per pass, no root-find), so the floor
                # can be relaxed to 0.001 (99.9% cap) to allow higher recycle
                # at large flows. If convergence degrades near the limit, the
                # Wegstein count in the solver summary will show it; raise this
                # back toward 0.01 if so.
                _makeup_floor   = 1.0     # kg/hr fresh solvent (negligible at
                #                           large flows; not the binding limit)
                _MIN_PURGE_FRAC = 0.001   # always purge >= 0.1% (recycle <= 99.9%)

                # Live progress for the recycle solve (Wegstein count).
                _recycle_progress = {'wegstein': 0}
                _recycle_status = st.empty() if display else _NoopUI()

                def _update_recycle_status():
                    w = _recycle_progress['wegstein']
                    _recycle_status.info(
                        f"\U0001f504 Solving recycle loop - "
                        f"{w} Wegstein iteration(s) ...")

                # -- Recycle-split specification -----------------------
                # S1.split is NOT a free knob. Each Wegstein pass it is
                # computed so the recycle delivers as much recovered
                # solvent as is useful and allowed - never more:
                #   * makeup-floor bound : recycled solvent <= target -
                #     floor, so the fresh makeup stays >= floor and the
                #     total solvent reaching the extractor is pinned at
                #     target_solvent_flow. This is what stops feed-
                #     moisture water (or any non-solvent volatile) from
                #     accumulating without bound - the purge is forced
                #     to carry out whatever M2 holds above the target.
                #   * impurity-cap bound : recycled impurity <=
                #     x_max*target/(1-x_max), so the impurity mass
                #     fraction at the M1 outlet stays <= x_max.
                #   * availability bound : cannot recycle more solvent
                #     than M2 holds.
                # The binding (smallest) bound wins; the rest is purged.
                # Because the split is a mass balance, not a search, the
                # loop has a single physical steady state and cannot run
                # away - so the old brentq split-search is not needed.
                def adjust_recycle_split():
                    m2 = S1.ins[0]
                    F_m2 = m2.F_mass
                    if F_m2 <= 1e-9:
                        S1.split = 0.0
                        return
                    solv_m2 = m2.imass[solv]
                    if solv_m2 <= 1e-12:
                        S1.split = 0.0          # nothing worth recycling
                        return
                    f_solv = solv_m2 / F_m2

                    # 1. makeup-floor bound on recycled *solvent*
                    solv_cap_makeup = max(
                        0.0, target_solvent_flow - _makeup_floor)

                    # 2. impurity-cap bound. At the M1 outlet the mass
                    #    balance gives imp_frac = imp_in/(target+imp_in);
                    #    setting that = x_max => imp_in <=
                    #    x_max*target/(1-x_max). A uniform splitter
                    #    carries impurity and solvent in the M2 ratio
                    #    (1-f_solv):f_solv, so this maps to a solvent
                    #    bound.
                    if recycle_x_max < 1.0:
                        imp_in_max = (recycle_x_max * target_solvent_flow
                                      / (1.0 - recycle_x_max))
                    else:
                        imp_in_max = float('inf')
                    if f_solv >= 1.0 - 1e-12:
                        solv_cap_impurity = solv_m2   # ~no impurity
                    else:
                        solv_cap_impurity = (imp_in_max * f_solv
                                             / (1.0 - f_solv))

                    # 3. availability bound, take the binding one
                    solv_recycle = max(0.0, min(solv_cap_makeup,
                                                solv_cap_impurity,
                                                solv_m2))
                    split = solv_recycle / solv_m2
                    # clamp so the loop always keeps an exit
                    S1.split = min(max(split, 0.0),
                                   1.0 - _MIN_PURGE_FRAC)

                S1.add_specification(adjust_recycle_split, run=True)

                def adjust_makeup():
                    import math
                    _recycle_progress['wegstein'] += 1
                    _update_recycle_status()
                    r_solv = recycle.imass[solv]
                    if not math.isfinite(r_solv) or r_solv < 0:
                        r_solv = 0.0
                    solvent.imass[solv] = max(
                        _makeup_floor, target_solvent_flow - r_solv)

                M1.add_specification(adjust_makeup, run=True)

                # ── Prime the recycle-loop streams ───────────────────
                # Wegstein needs starting values for every stream in the
                # cycle. The `recycle` stream declaration gives the
                # back-edge guess; running the front-end units once
                # populates the forward-pass streams so the extractor
                # doesn't see empty inputs on iteration 1.
                _advance_progress()  # Stage 3: priming recycle loop
                adjust_makeup()
                S102._run()
                M1._run()
                U103._run()
                U105._run()
                if feed_dry_enable:
                    D101._run()
                    C101._run()
                S101._run()
                U101._run()
                U102._run()
                E201._run()
            else:
                _advance_progress()  # Stage 3 (no recycle): nothing to prime

            # Create system (auto-detects the recycle loop)
            sys = bst.System.from_units('sys', bst.main_flowsheet.unit)

            if enable_recycle:
                # With the recycle split derived from a mass balance
                # (adjust_recycle_split), the loop has a single physical
                # steady state, so Wegstein just needs room to converge
                # and no outer brentq split-search is needed.
                sys.molar_tolerance          = 1e-2
                sys.relative_molar_tolerance = 1e-3
                sys.method                   = 'wegstein'
                sys.maxiter                  = 200

                _advance_progress()  # Stage 4: solving recycle
                _recycle_spinner = (st.spinner("Solving recycle loop ...")
                                    if display else contextlib.nullcontext())
                with _recycle_spinner:
                    sys.simulate()

                    # Post-simulation sanity check. Hard physical ceiling
                    # on the recycle mixer M2 = recycle + purge:
                    #   * recycle <= target/(1-x_max)  (the split spec)
                    #   * purge   <= feed + makeup <= feedflow + target
                    # Exceeding it is non-physical and means the loop did
                    # not converge - fail loudly instead of costing a
                    # nonsense stream.
                    _F_m2 = M2.outs[0].F_mass
                    _inventory_bound = (
                        target_solvent_flow
                        / max(1e-3, 1.0 - recycle_x_max)
                        + target_solvent_flow + feedflow)
                    if _F_m2 > _inventory_bound:
                        raise RuntimeError(
                            f"Recycle loop did not reach a physical "
                            f"steady state: mixed_recycle = "
                            f"{_F_m2:,.0f} kg/hr exceeds the sanity "
                            f"bound of {_inventory_bound:,.0f} kg/hr. "
                            f"Check convergence (sys.maxiter), the "
                            f"feed-moisture balance, or recycle_x_max.")

                    recycle_optimal_split = (
                        float(S1.split[0])
                        if hasattr(S1.split, '__len__')
                        else float(S1.split))
                    recycle_solver_msg = (
                        f"\u2705 Recycle split solved from mass "
                        f"balance: {recycle_optimal_split:.5f}")
                    _recycle_status.empty()   # clear live progress line
                    _run_status_lines.append(
                        f"Recycle solved in "
                        f"{_recycle_progress['wegstein']} Wegstein "
                        f"iteration(s).")
            else:
                _advance_progress()  # Stage 4 (no recycle): single solve
                sys.simulate()
                recycle_optimal_split = None
                recycle_solver_msg = None

            # ── Evaporator solver diagnostics ────────────────────────
            # n_run : how many times E301._run() was called this simulation.
            #         This is the SPEED number -- the evaporator sits inside
            #         the V-bisection inside the Wegstein loop, so it is the
            #         product of all the nested iteration counts.
            # n_fb  : how many of those calls tripped a guard and fell back to
            #         a guessed / crude result instead of solving. This is the
            #         TRUST number. If n_fb > 0 the evaporator's duty, area and
            #         capital cost are not solved values, and the minimum
            #         selling price built on them is not reliable -- even
            #         though every stream will still look perfectly healthy.
            evap_n_run = E301.n_run
            evap_n_fb = E301.n_fb
            evap_shortcut = bool(getattr(E301, '_design_from_shortcut', False))
            evap_guard_counts = dict(getattr(E301, '_guard_counts', {}))
            evap_guard_reasons = dict(getattr(E301, '_guard_reasons', {}))

            if display:
                if evap_n_fb == 0:
                    _run_status_lines.append(
                        f"Evaporator solved cleanly on all "
                        f"{evap_n_run:,} `_run` call(s) — rigorous duty, "
                        f"area and cost.")
                # The guard-trip breakdown (when evap_n_fb > 0) is rendered at
                # the bottom of the output, grouped with the volatility-screen
                # and proxy-compound notes. The data needed for it is already
                # captured in evap_n_fb / evap_shortcut / evap_guard_counts /
                # evap_guard_reasons above.

            _advance_progress()  # Stage 5: techno-economic analysis
            #                       COUNT PROCESSING STEPS FOR LABOR CALCULATION
            # Define which unit types involve solids
            solid_handling_units = (units.ConveyingBelt, units.Shredder, SolidSolventExtractor, SolidCooler)
            steps_involving_solids = 0
            steps_not_involving_solids = 0
            solid_units_list = []
            non_solid_units_list = []

            for unit in sys.units:
                # Skip storage tanks
                if isinstance(unit, units.StorageTank):
                    continue
                # Check if unit handles solids
                elif isinstance(unit, solid_handling_units):
                    steps_involving_solids += 1
                    solid_units_list.append(unit.ID)
                else:
                    steps_not_involving_solids += 1
                    non_solid_units_list.append(unit.ID)

            # Calculate operators per shift using the formula:
            operators_per_shift = round(
                (6.29 + 31.7 * (steps_involving_solids) + 0.23 * (steps_not_involving_solids)) ** 0.5)
            # Calculate total labor cost
            hours_per_year = 24 * operating_days  # Total hours in a year for continuous operation
            labor_cost = operator_hourly_wage * operators_per_shift * hours_per_year * (1+supervision_factor)

            # TEA Class Definition
            class ExtractionTEA(bst.TEA):
                def __init__(self, system, IRR, duration, depreciation, income_tax,
                             operating_days, construction_schedule, startup_months,
                             startup_FOCfrac, startup_VOCfrac, startup_salesfrac,
                             WC_over_FCI, finance_interest, finance_years, finance_fraction,
                             lang_factor, maintenance, property_insurance, property_tax,
                             labor_cost, fringe_benefits, supplies, administration,
                             Additional_direct_costs, indirect_costs):
                    super().__init__(
                        system=system, IRR=IRR, duration=duration,
                        depreciation=depreciation, income_tax=income_tax,
                        operating_days=operating_days, lang_factor=lang_factor,
                        construction_schedule=construction_schedule,
                        startup_months=startup_months, startup_FOCfrac=startup_FOCfrac,
                        startup_VOCfrac=startup_VOCfrac, startup_salesfrac=startup_salesfrac,
                        WC_over_FCI=WC_over_FCI, finance_interest=finance_interest,
                        finance_years=finance_years, finance_fraction=finance_fraction
                    )

                    self.maintenance = maintenance
                    self.property_insurance = property_insurance
                    self.property_tax = property_tax
                    self.labor_cost = labor_cost
                    self.fringe_benefits = fringe_benefits
                    self.supplies = supplies
                    self.administration = administration
                    self.Additional_direct_costs = Additional_direct_costs
                    self.indirect_costs = indirect_costs

                def _DPI(self, installed_equipment_cost):
                    return installed_equipment_cost * (1 + self.Additional_direct_costs)

                def _TDC(self, DPI):
                    indirect = DPI * self.indirect_costs
                    return DPI + indirect

                def _FCI(self, TDC):
                    return TDC

                def _FOC(self, FCI):
                    maintenance_cost = self.maintenance * FCI
                    insurance_cost = self.property_insurance * FCI
                    tax_cost = self.property_tax * FCI
                    fringe_cost = self.fringe_benefits * self.labor_cost
                    supplies_cost = self.supplies * self.labor_cost
                    admin_cost = self.administration * self.labor_cost

                    return (maintenance_cost + insurance_cost + tax_cost +
                            self.labor_cost + fringe_cost + supplies_cost + admin_cost)


            # Create TEA
            tea = ExtractionTEA(
                system=sys,
                IRR=IRR,
                duration=(Yearstart, Yearstart + duration),
                depreciation=depreciation,
                income_tax=income_tax,
                operating_days=operating_days,
                construction_schedule=construction_schedule,
                startup_months=startup_months,
                startup_FOCfrac=startup_FOCfrac,
                startup_VOCfrac=startup_VOCfrac,
                startup_salesfrac=startup_salesfrac,
                WC_over_FCI=WC_over_FCI,
                finance_interest=finance_interest,
                finance_years=finance_years,
                finance_fraction=finance_fraction,
                lang_factor=None,
                maintenance=maintenance,
                property_insurance=property_insurance,
                property_tax=property_tax,
                labor_cost=labor_cost,
                fringe_benefits=fringe_benefits,
                supplies=supplies,
                administration=administration,
                Additional_direct_costs=Additional_direct_costs,
                indirect_costs=indirect_costs,
            )

            # Get costs
            dfc = tea.DPI
            fci = tea.FCI
            tci = tea.TCI

            # ── Preflight: locate any NaN/inf before solve_price ─────────
            # BioSTEAM's solve_price raises a bare "nan encountered in
            # cashflow array" that names neither the unit nor the quantity at
            # fault. The NaN almost always originates upstream — a zero/NaN
            # product flow, or a unit whose installed cost or utility duty came
            # out non-finite — so scan the obvious sources first and fail with
            # a message that says WHERE.
            import math as _math

            def _finite(x):
                try:
                    return _math.isfinite(float(x))
                except (TypeError, ValueError):
                    return True   # non-numeric -> not our concern here

            _nan_problems = []

            # 1. Product stream must carry finite, positive flow, or the price
            #    solve (sales / production) is ill-defined.
            _prod_F = cooled_product.F_mass
            if not _finite(_prod_F):
                _nan_problems.append(
                    "product stream 'cooled_product' has non-finite mass flow")
            elif _prod_F <= 0:
                _zero_prod = (
                    "product stream 'cooled_product' has zero mass flow — "
                    "nothing is being produced to price.")
                # The most likely recent cause: the volatility screen kept a
                # PRODUCT solute in the VLE (alpha >= ALPHA_MIN), so it left in
                # the evaporator condensate instead of the concentrate that
                # becomes product. Surface the locked/kept split so this is
                # checkable.
                if _volatility_screen_info is not None:
                    _kept_ids = [i for i, _ in _volatility_screen_info['kept']]
                    if _kept_ids:
                        _zero_prod += (
                            " Note: the volatility screen kept "
                            f"{len(_kept_ids)} solute(s) in the VLE "
                            f"({', '.join(_kept_ids[:6])}"
                            f"{' …' if len(_kept_ids) > 6 else ''}); if your "
                            "target product is among them it is leaving in the "
                            "evaporator condensate rather than the product. "
                            "Raise ALPHA_MIN or exempt the product solute.")
                    else:
                        _zero_prod += (
                            " (Check extraction recovery and the "
                            "dryer/evaporator splits.)")
                _nan_problems.append(_zero_prod)

            # 2. Per-unit installed cost and utility cost.
            for _u in sys.units:
                try:
                    if not _finite(_u.installed_cost):
                        _nan_problems.append(
                            f"unit '{_u.ID}' has non-finite installed cost")
                    if not _finite(_u.utility_cost):
                        _nan_problems.append(
                            f"unit '{_u.ID}' has non-finite utility cost")
                    for _hu in getattr(_u, 'heat_utilities', ()):
                        if _hu.duty is not None and not _finite(_hu.duty):
                            _nan_problems.append(
                                f"unit '{_u.ID}' has non-finite heat-utility "
                                f"duty")
                            break
                except Exception:
                    pass

            # 3. Aggregate TEA capital figures.
            for _label, _val in (('DPI', dfc), ('FCI', fci), ('TCI', tci)):
                if not _finite(_val):
                    _nan_problems.append(f"TEA {_label} is non-finite")

            if _nan_problems:
                _msg = ("Cannot solve product price — a non-finite value "
                        "reached the cash-flow array. Source(s):\n  - "
                        + "\n  - ".join(dict.fromkeys(_nan_problems)))
                if display:
                    st.error("❌ " + _msg)
                raise RuntimeError(_msg)

            # Calculate minimum selling price (on dried product)
            Product_price = tea.solve_price(cooled_product)

            # ── Compute scalar metrics (shared by single & sweep modes) ──
            maintenance_cost   = tea.maintenance        * fci
            insurance_cost     = tea.property_insurance * fci
            tax_cost           = tea.property_tax       * fci
            labor_cost_val     = tea.labor_cost
            fringe_cost        = tea.fringe_benefits * tea.labor_cost
            supplies_cost      = tea.supplies        * tea.labor_cost
            admin_cost         = tea.administration  * tea.labor_cost
            total_foc          = tea.FOC
            annual_material_cost = sum(stream.cost * tea.operating_hours
                                       for stream in sys.feeds)
            annual_utility_cost = sum(
                (unit.utility_cost or 0.0) * tea.operating_hours
                for unit in sys.units if hasattr(unit, 'utility_cost'))
            total_voc            = annual_material_cost + annual_utility_cost
            total_operating_cost = total_foc + total_voc

            residual_solvent_frac = (
                cooled_product.imass[solv] / cooled_product.F_mass
                if cooled_product.F_mass > 0 else 0.0)
            product_T_C = cooled_product.T - 273.15
            product_F_kghr = cooled_product.F_mass

            installed_equipment_total = sum(
                u.installed_cost for u in sys.units)

            scalars = {
                # Inputs echoed for traceability
                'feedstock':            selected_feedstock,
                'solvent':              solv,
                'reactor_type':         reactor_type,
                'heatutility':          heatutility,
                'pressure_mode':        pressure_mode,
                'depreciation':         depreciation,
                'feedflow_kg_hr':       feedflow,
                'solvflow_kg_hr':       solvflow,
                'ExtractT_C':           ExtractT,
                'extractt_hr':          extractt,
                'particle_radius_cm':   Particlesize,
                'feed_dry_enable':      feed_dry_enable,
                'feed_dry_moisture':    (feed_dry_moisture if feed_dry_enable
                                         else moisture),
                'feed_dry_T_C':         (feed_dry_T if feed_dry_enable
                                         else None),
                'feed_moisture_as_received': moisture,
                'feed_water_evaporated_kg_per_hr': (
                    float(feed_dryer_vapor.F_mass) if feed_dry_enable else 0.0),
                'feed_to_storage_kg_per_hr': float(S101.ins[0].F_mass),
                'evap_n_effects':       evap_n_effects,
                'evap_target_solids':   evap_target_solids,
                # Solver diagnostics -- see the block after sys.simulate().
                # evap_n_fb > 0 means the evaporator cost/duty on this row is
                # an estimate, not a solved value.
                'evap_n_run':           evap_n_run,
                'evap_n_fb':            evap_n_fb,
                'evap_sizing_shortcut': evap_shortcut,
                'evap_guard_trips':     ("; ".join(
                    f"{k}x{v}" for k, v in evap_guard_counts.items())
                    if evap_guard_counts else ""),
                'dryer_moisture':       dryer_moisture,
                'enable_recycle':       enable_recycle,
                'recycle_x_max':        recycle_x_max,
                'Yearstart':            Yearstart,
                'elec_price_USD_per_kWh': elec_price,
                'feed_price_USD_per_kg':  feed_price,
                'operator_wage_USD_per_hr': operator_hourly_wage,
                'IRR':                  IRR,
                'plant_life_yr':        duration,
                'operating_days':       operating_days,
                'income_tax':           income_tax,
                # Key outputs
                'product_price_USD_per_kg': Product_price,
                'TCI_USD':              tci,
                'FCI_USD':              fci,
                'DPI_USD':              dfc,
                'installed_equipment_cost_USD': installed_equipment_total,
                'FOC_USD_per_yr':       total_foc,
                'VOC_USD_per_yr':       total_voc,
                'total_opcost_USD_per_yr': total_operating_cost,
                'annual_material_cost_USD': annual_material_cost,
                'annual_utility_cost_USD':  annual_utility_cost,
                'labor_cost_USD_per_yr':    labor_cost_val,
                'operators_per_shift':      operators_per_shift,
                'cooled_product_kg_per_hr': product_F_kghr,
                'cooled_product_residual_solvent_frac': residual_solvent_frac,
                'cooled_product_T_C':       product_T_C,
            }

            # Recycle-specific metrics (only meaningful when recycle on)
            if enable_recycle:
                m1_out_for_scalars = M1.outs[0]
                _F_m1 = m1_out_for_scalars.F_mass
                _F_solv_in = m1_out_for_scalars.imass[solv]
                _imp_frac_scalar = (
                    (_F_m1 - _F_solv_in) / _F_m1 if _F_m1 > 0 else 0.0)
                _split_scalar = (float(S1.split[0])
                                 if hasattr(S1.split, '__len__')
                                 else float(S1.split))
                scalars.update({
                    'recycle_split':          _split_scalar,
                    'purge_fraction':         1.0 - _split_scalar,
                    'impurity_frac_at_inlet': _imp_frac_scalar,
                    'fresh_makeup_kg_hr':     solvent.F_mass,
                    'recycle_kg_hr':          recycle.F_mass,
                    'purge_kg_hr':            purge.F_mass,
                    'evap_condensate_kg_hr':  condensate.F_mass,
                    'dryer_vapor_kg_hr':      dryer_vapor.F_mass,
                })
            else:
                scalars.update({
                    'recycle_split':          None,
                    'purge_fraction':         None,
                    'impurity_frac_at_inlet': None,
                    'fresh_makeup_kg_hr':     solvent.F_mass,
                    'recycle_kg_hr':          None,
                    'purge_kg_hr':            None,
                    'evap_condensate_kg_hr':  condensate.F_mass,
                    'dryer_vapor_kg_hr':      dryer_vapor.F_mass,
                })

            # ── Dried-product (extract) composition ───────────────────
            # Emit one column per component so the sweep CSV captures the
            # full extract composition for every run. We report both the
            # absolute mass flow (kg/hr) and the concentration (mg/g),
            # mirroring the single-run "Dried Product Composition" table.
            # Components differing between runs (e.g. different feedstocks
            # or solvents) simply produce extra columns; pandas fills the
            # missing entries with NaN when the rows are combined.
            _comp_total_g = cooled_product.F_mass * 1000.0  # kg/hr → g/hr
            for _chem_id, _mass_flow in zip(cooled_product.chemicals.IDs,
                                            cooled_product.mass):
                if _mass_flow > 1e-6:
                    _orig = ts_to_original(_chem_id)
                    _mg_g = ((_mass_flow * 1e6) / _comp_total_g
                             if _comp_total_g > 0 else 0.0)
                    scalars[f'extract_{_orig}_kg_per_hr'] = float(_mass_flow)
                    scalars[f'extract_{_orig}_mg_per_g'] = float(_mg_g)

            # In sweep mode the per-run display is skipped — only the
            # scalar dict matters.
            if not display:
                return scalars

            # ── Temperature-exposure flags ────────────────────────────
            # Flag any handled compound held above its reported thermal-
            # degradation threshold. Heating columns (water / other solvent)
            # apply to the extractor AND evaporator; drying applies to SD1.
            degradation_flags = []
            degradation_unclassified = []
            degradation_info = {}
            try:
                _heat_col = ('water' if str(solv).lower() == 'water'
                             else 'other_solvent')
                _T_extract_C = float(ExtractT)
                _solv_chem = tmo.settings.chemicals[solv]
                try:
                    _T_evap_C = float(_solv_chem.Tsat(max(evap_P)) - 273.15)
                except Exception:
                    _T_evap_C = float(_solv_chem.Tb - 273.15)
                _T_dry_C = float(SD1.T - 273.15)
                _T_feed_dry_C = (float(D101.T - 273.15)
                                 if feed_dry_enable else None)
                degradation_info = {
                    'heating_column': _heat_col,
                    'T_extractor_C': round(_T_extract_C, 1),
                    'T_evaporator_C': round(_T_evap_C, 1),
                    'T_dryer_C': round(_T_dry_C, 1),
                }
                if _T_feed_dry_C is not None:
                    degradation_info['T_feed_dryer_C'] = round(_T_feed_dry_C, 1)

                _chem_meta = build_chem_meta(feedchem_entries)

                def _records_in(stream):
                    recs, seen = [], set()
                    for _tsid, _m in zip(stream.chemicals.IDs, stream.mass):
                        if _m <= 1e-6 or str(_tsid).lower() == str(solv).lower():
                            continue
                        _orig = ts_to_original(_tsid)
                        _key = _tt_norm(_orig)
                        if _key in seen:
                            continue
                        seen.add(_key)
                        _meta = _chem_meta.get(_key, {})
                        recs.append({'name': _orig, 'cas': _meta.get('cas'),
                                     'chebi': _meta.get('chebi')})
                    return recs

                _exposures = []
                # The feedstock drum dryer sees every compound in the raw
                # feed, at the drying temperature — so thermolabile targets
                # can be lost before extraction even starts.
                if feed_dry_enable and D101.outs[0].F_mass > 1e-9:
                    _exposures.append(
                        {'unit': 'Feedstock drum dryer (D101)',
                         'column': 'drying',
                         'temp_C': _T_feed_dry_C,
                         'chemicals': _records_in(feed)})
                _exposures += [
                    {'unit': 'Extractor (E201)',  'column': _heat_col,
                     'temp_C': _T_extract_C, 'chemicals': _records_in(feed)},
                    {'unit': 'Evaporator (E301)', 'column': _heat_col,
                     'temp_C': _T_evap_C,    'chemicals': _records_in(extract)},
                    {'unit': 'Spray dryer (SD1)', 'column': 'drying',
                     'temp_C': _T_dry_C,     'chemicals': _records_in(E301.outs[0])},
                ]
                _chebi = ChebiResolver(cache_path=os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    'chebi_group_cache.json'))
                degradation_info['chebi_available'] = _chebi.available
                _classifier = ThresholdClassifier(chebi=_chebi)
                degradation_flags, degradation_unclassified = \
                    evaluate_temperature_flags(_exposures, _classifier)
                _chebi.save_cache()
                degradation_info['chebi_error'] = _chebi.error
            except Exception as _exc:
                degradation_info['error'] = str(_exc)

            # Display Results
            _advance_progress()  # Stage 6: preparing results
            _finish_progress()
            st.success("✅ Simulation completed successfully!")

            # Map each thermally-flagged compound -> the stage(s) where it
            # exceeded a threshold. Used to mark the dried-product composition.
            def _stage_label(u):
                ul = str(u).lower()
                if 'extract' in ul:
                    return 'Extractor'
                if 'evapor' in ul:
                    return 'Evaporator'
                if 'dry' in ul:
                    return 'Dryer'
                return str(u)
            _flag_note = {}
            for _f in degradation_flags:
                _flag_note.setdefault(_tt_norm(_f.chemical), set()).add(
                    _stage_label(_f.unit))

            # Create tabs for organized output
            tab1, tab2, tab3, tab4, tab5 = st.tabs(
                ["📊 Summary", "💰 Capital Costs", "🔄 Operating Costs",
                 "📈 Stream Data", "🌡️ Degradation Flags"])

            with tab1:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Direct Fixed Capital (DFC)", f"${dfc:,.0f}")
                with col2:
                    st.metric("Fixed Capital Investment (FCI)", f"${fci:,.0f}")
                with col3:
                    st.metric("Total Capital Investment (TCI)", f"${tci:,.0f}")

                st.markdown("---")

                # Extract composition
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("📦 Dried Product Composition")

                    # Create composition table
                    comp_data = []
                    total_mass = cooled_product.F_mass
                    total_mass_g = total_mass * 1000  # kg/hr to g/hr

                    # Build a lookup of feed mass flows by thermosteam ID for % of feed calc
                    feed_mass_lookup = {chem: mf for chem, mf in zip(feed.chemicals.IDs, feed.mass) if mf > 0}

                    for chemical, mass_flow in zip(cooled_product.chemicals.IDs, cooled_product.mass):
                        if mass_flow > 1e-6:  # Only show components with significant amounts
                            mg_per_g = (mass_flow * 1e6) / total_mass_g if total_mass_g > 0 else 0
                            original_name = ts_to_original(chemical)
                            feed_mass = feed_mass_lookup.get(chemical, 0)
                            pct_feed = (mass_flow / feed_mass * 100) if feed_mass > 0 else 0
                            note = _flag_note.get(_tt_norm(original_name))
                            comp_data.append({
                                'Chemical': (("❗ " + original_name)
                                             if note else original_name),
                                'Mass Flow (kg/hr)': f"{mass_flow:.4f}",
                                'Concentration (mg/g)': _fmt_mg_g(mg_per_g),
                                '% of Feed': f"{pct_feed:.2f}%",
                                'Degradation risk': ("⚠️ " + ", ".join(sorted(note))
                                                     if note else ""),
                            })

                    if comp_data:
                        df_comp = pd.DataFrame(comp_data)
                        has_flag = df_comp['Degradation risk'].astype(bool).any()
                        if not has_flag:
                            df_comp = df_comp.drop(columns=['Degradation risk'])
                        try:
                            if has_flag:
                                styled = df_comp.style.apply(
                                    lambda r: ['background-color: #fde2e1'
                                               if r['Degradation risk'] else ''
                                               ] * len(r), axis=1)
                                st.dataframe(styled, use_container_width=True,
                                             hide_index=True)
                            else:
                                st.dataframe(df_comp, use_container_width=True,
                                             hide_index=True)
                        except Exception:
                            st.dataframe(df_comp, use_container_width=True,
                                         hide_index=True)

                        if has_flag:
                            st.caption(
                                "❗ / highlighted rows are compounds held above a "
                                "thermal-degradation threshold in the stage(s) noted "
                                "under “Degradation risk” (extractor → evaporator → "
                                "dryer). Real dried product would be expected to "
                                "contain lower quantities of these than shown. See "
                                "the 🌡️ Degradation Flags tab for temperatures and "
                                "limits.")

                        st.write(f"**Cooled Product Flow:** {cooled_product.F_mass:.2f} kg/hr")
                        dsf = cooled_product.imass[solv] / cooled_product.F_mass if cooled_product.F_mass > 0 else 0
                        st.write(f"**Residual Solvent:** {dsf:.1%}")
                        st.write(f"**Product Temperature:** {cooled_product.T - 273.15:.1f} °C")
                    else:
                        st.info("No dried product composition data available")

                with col2:
                    st.subheader("🎯 Minimum Product Selling Price")
                    st.metric("Price per kg", f"${Product_price:.4f} USD/kg",
                              help="The minimum price needed to achieve the target IRR")

                    # Add key process parameters
                    st.markdown("#### Key Process Parameters")
                    st.write(f"**Feedstock:** {selected_feedstock}")
                    st.write(f"**Solvent:** {solv}")
                    st.write(f"**Extraction Temp:** {ExtractT}°C")
                    st.write(f"**Extraction Time:** {extractt} hr")
                    st.write(f"**Pressure Mode:** {pressure_mode}")

                st.markdown("---")
                try:
                    diagram_path = 'extraction_system_diagram'
                    sys.diagram(file=diagram_path, format='png')
                    img = Image.open(f'{diagram_path}.png')
                    st.image(img, caption="Process Flow Diagram", use_container_width=True)
                except Exception as e:
                    st.info("Process diagram could not be generated. Graphviz may not be installed.")

            with tab2:
                st.subheader("💰 Capital Cost Breakdown")

                # Prepare capital cost data
                cost_items = []

                for unit in sys.units:
                    if unit.installed_cost > 0:
                        percentage = (unit.installed_cost / tci) * 100
                        cost_items.append({
                            'Item': unit.ID,
                            'Cost': unit.installed_cost,
                            'Percentage': percentage,
                            'Type': 'Equipment'
                        })

                installed_equipment_total = sum(unit.installed_cost for unit in sys.units)
                additional_direct = installed_equipment_total * tea.Additional_direct_costs
                if additional_direct > 0:
                    cost_items.append({
                        'Item': 'Additional Direct Costs',
                        'Cost': additional_direct,
                        'Percentage': (additional_direct / tci) * 100,
                        'Type': 'Additional'
                    })

                indirect_costs_amount = tea.DPI * tea.indirect_costs
                if indirect_costs_amount > 0:
                    cost_items.append({
                        'Item': 'Indirect Costs',
                        'Cost': indirect_costs_amount,
                        'Percentage': (indirect_costs_amount / tci) * 100,
                        'Type': 'Additional'
                    })

                working_capital = tea.WC_over_FCI * fci
                if working_capital > 0:
                    cost_items.append({
                        'Item': 'Working Capital',
                        'Cost': working_capital,
                        'Percentage': (working_capital / tci) * 100,
                        'Type': 'Additional'
                    })

                # Create DataFrame and sort
                df_capital = pd.DataFrame(cost_items)
                df_capital = df_capital.sort_values('Percentage', ascending=False)
                df_capital['Cost'] = df_capital['Cost'].apply(lambda x: f"${x:,.0f}")
                df_capital['Percentage'] = df_capital['Percentage'].apply(lambda x: f"{x:.2f}%")

                st.dataframe(df_capital, use_container_width=True, hide_index=True)

                # Pie chart for all capital costs - showing each item individually
                fig, ax = plt.subplots(figsize=(12, 8))
                pie_data = []
                pie_labels = []

                # Add all costs individually (both equipment and additional)
                for item in cost_items:
                    pie_data.append(item['Cost'])
                    pie_labels.append(item['Item'])

                if pie_data:
                    colors = plt.cm.Set3(range(len(pie_data)))
                    wedges, texts, autotexts = ax.pie(pie_data, labels=pie_labels, autopct='%1.1f%%',
                                                      startangle=90, colors=colors, pctdistance=0.85)

                    # Make percentage text more readable
                    for autotext in autotexts:
                        autotext.set_color('white')
                        autotext.set_fontweight('bold')
                        autotext.set_fontsize(9)

                    # Make labels more readable
                    for text in texts:
                        text.set_fontsize(9)

                    ax.set_title('Total Capital Investment (TCI) Breakdown', fontsize=14, fontweight='bold')
                    st.pyplot(fig)

                # Show summary totals
                st.markdown("---")
                col1, col2 = st.columns(2)
                with col1:
                    total_equipment = sum(item['Cost'] for item in cost_items if item['Type'] == 'Equipment')
                    st.metric("Total Equipment Costs", f"${total_equipment:,.0f}")
                with col2:
                    total_additional = sum(item['Cost'] for item in cost_items if item['Type'] == 'Additional')
                    st.metric("Total Additional Costs", f"${total_additional:,.0f}")

            with tab3:
                st.subheader("🔄 Annual Operating Costs")

                # Calculate operating costs
                maintenance_cost = tea.maintenance * fci
                insurance_cost = tea.property_insurance * fci
                tax_cost = tea.property_tax * fci
                labor_cost_val = tea.labor_cost
                fringe_cost = tea.fringe_benefits * tea.labor_cost
                supplies_cost = tea.supplies * tea.labor_cost
                admin_cost = tea.administration * tea.labor_cost

                total_foc = tea.FOC

                annual_material_cost = sum(stream.cost * tea.operating_hours for stream in sys.feeds)
                # Some units return None for utility_cost when their feed
                # is effectively empty (degenerate sizing); treat as zero.
                annual_utility_cost = sum((unit.utility_cost or 0.0) * tea.operating_hours
                                          for unit in sys.units if hasattr(unit, 'utility_cost'))

                total_voc = annual_material_cost + annual_utility_cost
                total_operating_cost = total_foc + total_voc

                # Create operating costs DataFrame
                operating_items = [
                    {'Item': 'Maintenance', 'Cost': maintenance_cost, 'Type': 'Fixed'},
                    {'Item': 'Property Insurance', 'Cost': insurance_cost, 'Type': 'Fixed'},
                    {'Item': 'Property Tax', 'Cost': tax_cost, 'Type': 'Fixed'},
                    {'Item': 'Labor', 'Cost': labor_cost_val, 'Type': 'Fixed'},
                    {'Item': 'Fringe Benefits', 'Cost': fringe_cost, 'Type': 'Fixed'},
                    {'Item': 'Supplies', 'Cost': supplies_cost, 'Type': 'Fixed'},
                    {'Item': 'Administration', 'Cost': admin_cost, 'Type': 'Fixed'},
                    {'Item': 'Raw Materials', 'Cost': annual_material_cost, 'Type': 'Variable'},
                    {'Item': 'Utilities', 'Cost': annual_utility_cost, 'Type': 'Variable'},
                ]

                df_operating = pd.DataFrame(operating_items)
                df_operating['Percentage'] = (df_operating['Cost'] / total_operating_cost * 100).apply(
                    lambda x: f"{x:.2f}%")
                df_operating = df_operating.sort_values('Cost', ascending=False)
                df_operating['Cost'] = df_operating['Cost'].apply(lambda x: f"${x:,.0f}")

                st.dataframe(df_operating, use_container_width=True, hide_index=True)

                # Pie chart for operating costs - clean design with initials only
                fig, ax = plt.subplots(figsize=(10, 8))

                # Get the actual numeric values from operating_items
                pie_data_values = []
                pie_labels_initials = []
                legend_items = []
                colors = []

                # Create mapping for initials
                initials_map = {
                    'Maintenance': 'M',
                    'Property Insurance': 'PI',
                    'Property Tax': 'PT',
                    'Labor': 'L',
                    'Fringe Benefits': 'FB',
                    'Supplies': 'S',
                    'Administration': 'A',
                    'Raw Materials': 'RM',
                    'Utilities': 'U'
                }

                for item in operating_items:
                    pie_data_values.append(item['Cost'])

                    # Get initial for this item
                    initial = initials_map.get(item['Item'], item['Item'][:2].upper())
                    pie_labels_initials.append(initial)

                    # Create legend entry
                    type_label = "Fixed" if item['Type'] == 'Fixed' else "Variable"
                    legend_items.append(f"{initial} = {item['Item']} ({type_label})")

                    if item['Type'] == 'Fixed':
                        colors.append('lightblue')
                    else:
                        colors.append('lightcoral')

                # Create pie chart with initials outside, NO percentages
                wedges, texts = ax.pie(pie_data_values,
                                       labels=pie_labels_initials,
                                       startangle=90,
                                       colors=colors,
                                       labeldistance=1.15,
                                       wedgeprops={'edgecolor': 'black', 'linewidth': 1.5})

                # Make label text (initials) more readable
                for text in texts:
                    text.set_fontsize(12)
                    text.set_fontweight('bold')

                ax.set_title('Annual Operating Costs Breakdown', fontsize=14, fontweight='bold', pad=20)

                # Create legend below the chart
                from matplotlib.patches import Patch

                legend_elements = []
                for i, legend_text in enumerate(legend_items):
                    legend_elements.append(Patch(facecolor=colors[i], label=legend_text))

                ax.legend(handles=legend_elements,
                          loc='upper center',
                          bbox_to_anchor=(0.5, -0.05),
                          ncol=3,
                          fontsize=10,
                          frameon=True,
                          title="Cost Categories")

                plt.tight_layout()
                st.pyplot(fig)

                st.markdown("---")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Fixed Operating Costs", f"${total_foc:,.0f}")
                with col2:
                    st.metric("Total Variable Operating Costs", f"${total_voc:,.0f}")
                with col3:
                    st.metric("Total Operating Costs", f"${total_operating_cost:,.0f}")

            with tab4:
                st.subheader("📈 Stream Information")

                # ── Solvent recycle summary ───────────────────────────
                # Mirrors the [Specifications check] / [Recycle / purge
                # sizing] / [Overall mass balance] block from the
                # recycle_test reference script.
                if enable_recycle:
                    st.markdown("### ♻️ Solvent Recycle Summary")
                    if recycle_solver_msg:
                        st.info(recycle_solver_msg)

                    m1_out = M1.outs[0]
                    F_m1     = m1_out.F_mass
                    F_solv   = m1_out.imass[solv]
                    impurity_mass = F_m1 - F_solv
                    imp_frac = impurity_mass / F_m1 if F_m1 > 0 else 0.0

                    F_recycle = recycle.F_mass
                    F_makeup  = solvent.F_mass    # fresh makeup
                    F_purge   = purge.F_mass
                    F_evap    = condensate.F_mass
                    F_dryer   = dryer_vapor.F_mass
                    F_m2      = M2.outs[0].F_mass

                    cap_pass = imp_frac <= recycle_x_max + 1e-6
                    cap_status = "✅ PASS" if cap_pass else "❌ FAIL"

                    # Specifications check — top-level metrics
                    sc1, sc2, sc3 = st.columns(3)
                    with sc1:
                        st.metric(
                            f"{solv} delivered to extractor",
                            f"{F_solv:,.2f} kg/hr",
                            delta=f"target {solvflow:.2f} kg/hr",
                            delta_color="off",
                        )
                    with sc2:
                        st.metric(
                            "Impurity mass fraction at M1 outlet",
                            f"{imp_frac:.5f}",
                            delta=f"cap {recycle_x_max:.5f} ({cap_status})",
                            delta_color="off",
                        )
                    with sc3:
                        split_val = (float(S1.split[0])
                                     if hasattr(S1.split, '__len__')
                                     else float(S1.split))
                        st.metric(
                            "Recycle split (S1)",
                            f"{split_val:.4f}",
                            delta=f"purge fraction "
                                  f"{(1.0 - split_val) * 100:.2f} %",
                            delta_color="off",
                        )

                    # Impurity components present at M1 outlet
                    present_impurities = [
                        c for c in m1_out.chemicals.IDs
                        if c != solv and m1_out.imass[c] > 1e-9
                    ]
                    if present_impurities:
                        imp_rows = [
                            {
                                'Component': ts_to_original(c),
                                'Mass flow (kg/hr)':
                                    f"{m1_out.imass[c]:.4f}",
                                'Mass fraction':
                                    f"{m1_out.imass[c] / F_m1:.5f}"
                                    if F_m1 > 0 else "0",
                            }
                            for c in present_impurities
                        ]
                        with st.expander(
                                "Impurity components at solvent inlet "
                                "(M1 outlet)"):
                            st.dataframe(pd.DataFrame(imp_rows),
                                         use_container_width=True,
                                         hide_index=True)

                    # Recycle / purge sizing table
                    sizing_rows = [
                        {'Quantity': f'Recycle flow (S1 → M1)',
                         'Value': f'{F_recycle:,.2f} kg/hr'},
                        {'Quantity': f'Purge flow (S1 → out)',
                         'Value': f'{F_purge:,.2f} kg/hr'},
                        {'Quantity': f'Fresh {solv} makeup',
                         'Value': f'{F_makeup:,.2f} kg/hr'},
                        {'Quantity':
                            'Evap. condensate contribution to M2',
                         'Value':
                            f'{F_evap:,.2f} kg/hr  '
                            f'({100 * F_evap / F_m2 if F_m2 else 0:5.1f} %)'},
                        {'Quantity':
                            'Dryer-vapor (condensed) contribution to M2',
                         'Value':
                            f'{F_dryer:,.2f} kg/hr  '
                            f'({100 * F_dryer / F_m2 if F_m2 else 0:5.1f} %)'},
                    ]
                    st.markdown("**Recycle / purge sizing**")
                    st.dataframe(pd.DataFrame(sizing_rows),
                                 use_container_width=True, hide_index=True)

                    # Overall mass balance: in = feed + makeup ;
                    # out = feed-dryer vapor + solid_residual
                    #       + cooled_product + purge
                    F_feed_vap = (feed_dryer_vapor.F_mass
                                  if feed_dry_enable else 0.0)
                    F_in  = feed.F_mass + F_makeup
                    F_out = (F_feed_vap + solid_residual.F_mass
                             + cooled_product.F_mass + F_purge)
                    closure_err = F_in - F_out
                    st.markdown("**Overall mass balance**")
                    mb_rows = [
                        {'Stream': 'IN  : feed + fresh makeup',
                         'kg/hr': f'{F_in:,.3f}'},
                        {'Stream': 'OUT : feed-dryer vapor',
                         'kg/hr': f'{F_feed_vap:,.3f}'},
                        {'Stream': 'OUT : residual + dried product + purge',
                         'kg/hr': f'{F_out - F_feed_vap:,.3f}'},
                        {'Stream': 'Closure error',
                         'kg/hr': f'{closure_err:+.3e}'},
                    ]
                    st.dataframe(pd.DataFrame(mb_rows),
                                 use_container_width=True, hide_index=True)

                    st.markdown("---")

                # Display key streams
                streams_to_display = [feed]
                if feed_dry_enable:
                    streams_to_display += [feed_dryer_vapor, cooled_feed]
                streams_to_display += [solvent, solid_residual,
                                       extract, condensate, dried_product,
                                       cooled_product]
                if enable_recycle:
                    # Insert recycle-loop streams so the user can inspect
                    # composition at every node of the loop.
                    streams_to_display += [dryer_vapor, condensed_dryer_vapor,
                                           mixed_recycle, recycle, purge,
                                           M1.outs[0]]

                # Compounds flagged for thermal degradation -> the stage(s)
                # where they exceeded a threshold, so they can be marked in the
                # composition tables below.
                def _stage_label(u):
                    ul = str(u).lower()
                    if 'feed' in ul and 'dry' in ul:
                        return 'Feed dryer'
                    if 'extract' in ul:
                        return 'Extractor'
                    if 'evapor' in ul:
                        return 'Evaporator'
                    if 'dry' in ul:
                        return 'Dryer'
                    return str(u)
                _flag_note = {}
                for _f in degradation_flags:
                    _flag_note.setdefault(_tt_norm(_f.chemical), set()).add(
                        _stage_label(_f.unit))

                for stream in streams_to_display:
                    with st.expander(f"Stream: {stream.ID}"):
                        stream_data = {
                            'Property': ['Total Flow (kg/hr)', 'Temperature (°C)', 'Pressure (Pa)'],
                            'Value': [
                                f"{stream.F_mass:.2f}",
                                f"{stream.T - 273.15:.2f}",
                                f"{stream.P:.0f}"
                            ]
                        }
                        st.table(pd.DataFrame(stream_data))

                        # Show composition in mg/g
                        if stream.F_mass > 1e-6:
                            comp_data = []
                            total_mass_g = stream.F_mass * 1000  # kg/hr to g/hr
                            for chemical, mass_flow in zip(stream.chemicals.IDs, stream.mass):
                                if mass_flow > 1e-9:
                                    original_name = ts_to_original(chemical)
                                    mg_per_g = (mass_flow * 1e6) / total_mass_g  # mg/g of stream
                                    comp_data.append({
                                        'Chemical': original_name,
                                        'Mass Flow (kg/hr)': f"{mass_flow:.6f}",
                                        'Concentration (mg/g)': _fmt_mg_g(mg_per_g),
                                    })
                            if comp_data:
                                st.write("**Composition:**")
                                st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)

            with tab5:
                st.subheader("🌡️ Thermal Degradation Risk Flags")
                if degradation_info.get('error'):
                    st.error("Could not evaluate temperature flags: "
                             + degradation_info['error'])
                else:
                    _col = degradation_info.get('heating_column', 'water')
                    _col_label = ('Heating (Water)' if _col == 'water'
                                  else 'Heating (Other Solvent)')
                    _T_fd = degradation_info.get('T_feed_dryer_C')
                    if _T_fd is None:
                        c1, c2, c3 = st.columns(3)
                        c0 = None
                    else:
                        c0, c1, c2, c3 = st.columns(4)
                    if c0 is not None:
                        with c0:
                            st.metric("Feed-dryer exposure",
                                      f"{_T_fd:.1f} °C",
                                      help="Drum dryer on the as-received "
                                           "feedstock, upstream of storage. "
                                           "Checked against the “Drying” column")
                    with c1:
                        st.metric("Extractor exposure",
                                  f"{degradation_info.get('T_extractor_C', 0):.1f} °C",
                                  help=f"Checked against the “{_col_label}” column")
                    with c2:
                        st.metric("Evaporator exposure (hottest effect)",
                                  f"{degradation_info.get('T_evaporator_C', 0):.1f} °C",
                                  help=f"Checked against the “{_col_label}” column")
                    with c3:
                        st.metric("Product-dryer exposure",
                                  f"{degradation_info.get('T_dryer_C', 0):.1f} °C",
                                  help="Checked against the “Drying” column")

                    if degradation_info.get('chebi_available') is False:
                        st.info("ChEBI lookup was unavailable, so compound-group "
                                "matching was skipped (exact-name matches still "
                                "applied). To enable group flags without FTP, "
                                "download `chebi_lite.obo` over HTTPS from "
                                "https://ftp.ebi.ac.uk/pub/databases/chebi/ontology/chebi_lite.obo "
                                "and place it next to the scripts (or set "
                                "`CHEBI_OBO_PATH`); alternatively "
                                "`pip install libchebipy`.")
                    if degradation_info.get('chebi_error'):
                        st.caption("ChEBI note: " + str(degradation_info['chebi_error']))

                    if degradation_flags:
                        n_comp = len({f.chemical for f in degradation_flags})
                        st.warning(f"⚠️ {n_comp} compound(s) were held above a "
                                   f"temperature known to cause degradation.")
                        for _sentence in narrative_lines(degradation_flags):
                            st.markdown("- " + _sentence)
                        _df_flags = pd.DataFrame(flags_to_rows(degradation_flags))
                        _df_flags = _df_flags[[
                            'chemical', 'unit', 'column_label', 'exposure_C',
                            'threshold_C', 'exceedance_C', 'governed_by',
                            'matched_categories', 'match_basis']]
                        _df_flags = _df_flags.rename(columns={
                            'chemical': 'Compound', 'unit': 'Unit',
                            'column_label': 'Threshold basis',
                            'exposure_C': 'Exposure (°C)',
                            'threshold_C': 'Threshold (°C)',
                            'exceedance_C': 'Over by (°C)',
                            'governed_by': 'Governing category',
                            'matched_categories': 'All matched categories',
                            'match_basis': 'Match'})
                        _df_flags = _df_flags.sort_values(
                            'Over by (°C)', ascending=False)
                        with st.expander("Detailed flag table", expanded=False):
                            st.dataframe(_df_flags, use_container_width=True,
                                         hide_index=True)
                    else:
                        st.success("✅ No handled compound was held above its "
                                   "thermal-degradation threshold.")

                    if degradation_unclassified:
                        with st.expander(
                                f"Compounds with no threshold-table match "
                                f"({len(degradation_unclassified)})",
                                expanded=False):
                            st.caption("These handled compounds matched no named "
                                       "row and no ChEBI group, so they were not "
                                       "checked.")
                            st.write(", ".join(degradation_unclassified))

            # Solver summary — the recycle/evaporator convergence details,
            # rendered once here at the bottom instead of stacking green
            # banners at the top and pushing the results down. (The overall
            # "Simulation completed successfully" confirmation stays inline,
            # just above the results.)
            if display and _run_status_lines:
                st.success(
                    "✅ **Solver summary**\n\n"
                    + "\n".join(f"- {line}" for line in _run_status_lines))

            # Guard trip breakdown (only when the evaporator tripped a guard) —
            # grouped here at the bottom with the other diagnostics.
            if display and evap_n_fb > 0:
                _pct = 100.0 * evap_n_fb / max(1, evap_n_run)
                with st.expander("⚠️ Evaporator guard trip breakdown",
                                 expanded=False):
                    if evap_shortcut:
                        st.warning(
                            f"⚠️ Evaporator tripped a guard on "
                            f"**{evap_n_fb:,}** of **{evap_n_run:,}** "
                            f"`_run` call(s) ({_pct:.1f}%), **including "
                            f"the final one** — so its sizing came from "
                            f"the **short-cut energy balance**, not the "
                            f"rigorous cascade. Area/cost are a defensible "
                            f"estimate (validated to within ~3–8% at high "
                            f"V, erring high), but they are not solved "
                            f"values.")
                    else:
                        st.info(
                            f"ℹ️ Evaporator tripped a guard on "
                            f"**{evap_n_fb:,}** of **{evap_n_run:,}** "
                            f"`_run` call(s) ({_pct:.1f}%) during "
                            f"convergence, but **not on the final one** — "
                            f"the reported sizing is from the rigorous "
                            f"cascade. Transient trips on intermediate "
                            f"iterates are not fatal, but they indicate "
                            f"the flash is near its limits.")
                    st.dataframe(
                        pd.DataFrame(
                            [{'guard': k, 'trips': v}
                             for k, v in sorted(
                                 evap_guard_counts.items(),
                                 key=lambda kv: -kv[1])],
                        ),
                        use_container_width=True, hide_index=True)
                    for _g, _rs in evap_guard_reasons.items():
                        st.caption(f"**{_g}** — distinct reasons:")
                        st.code("\n".join(_rs))

            # Volatility screen (computed earlier; rendered here so it sits
            # at the bottom with the other chemistry notes).
            if display and _volatility_screen_info is not None:
                _vsi = _volatility_screen_info
                _kept, _locked = _vsi['kept'], _vsi['locked']
                with st.expander("🧪 Volatility screen results",
                                 expanded=False):
                    st.caption(
                        f"{len(_kept)} solute(s) kept in VLE, {len(_locked)} "
                        f"locked to liquid — alpha vs {_vsi['solv']} at "
                        f"{_vsi['T_screen_C']:.0f} °C, cutoff "
                        f"{_vsi['alpha_min']:g}.")
                    st.caption(
                        "Solutes KEPT in the VLE co-evaporate with the solvent "
                        "and reach the condensate — they are what "
                        "`recycle_x_max` caps. Solutes LOCKED to liquid are "
                        "treated as non-volatile and report entirely to the "
                        "concentrate. If a compound you expect to co-distil "
                        "appears in the locked list, lower ALPHA_MIN; if the "
                        "evaporator flash is driving the temperature out of "
                        "range, raise it.")
                    _rows = ([{'chemical': _id, 'alpha': f"{_a:.3g}",
                               'in VLE': '✅ yes'} for _id, _a in
                              sorted(_kept, key=lambda kv: -kv[1])]
                             + [{'chemical': _id,
                                 'alpha': ('n/a (Psat unevaluable)'
                                           if _a != _a else f"{_a:.3g}"),
                                 'in VLE': '— locked'} for _id, _a in
                                sorted(_locked,
                                       key=lambda kv: (kv[1] != kv[1],
                                                       -(kv[1] if kv[1] == kv[1]
                                                         else 0)))])
                    if _rows:
                        st.dataframe(pd.DataFrame(_rows),
                                     use_container_width=True, hide_index=True)
                    if not _kept:
                        st.warning(
                            "⚠️ No solute is volatile enough to enter the "
                            "vapour, so the condensate is essentially pure "
                            "solvent and the impurity-cap bound on the recycle "
                            "split will never bind — `recycle_x_max` has no "
                            "effect on this run. Lower ALPHA_MIN if you expect "
                            "solute carryover.")

            # Show proxy warnings at bottom if any
            if proxy_warnings:
                with st.expander("⚠️ Proxy Compound Warnings", expanded=False):
                    for w in proxy_warnings:
                        st.warning(w)

            return scalars

        except Exception as e:
            if display:
                st.error(f"❌ An error occurred during simulation: {str(e)}")
                st.exception(e)
                return None
            # In sweep mode the outer loop catches and records the error.
            raise


# ============================================================================
# Dispatch — Single Run or Parameter Sweep
# ============================================================================

def _collect_base_params():
    """Snapshot the current sidebar widget values into a single dict.

    These are the values used directly in Single Run mode, and the
    "constant" values for any non-swept parameter in Parameter Sweep
    mode (swept values override them per run)."""
    return {
        'selected_feedstock':       selected_feedstock,
        'feedflow':                 feedflow,
        'solv':                     solv,
        'solvflow':                 solvflow,
        'solv_flow_mode':           solv_flow_mode,
        'solv_feed_ratio':          solv_feed_ratio,
        'ExtractT':                 ExtractT,
        'extractt':                 extractt,
        'Particlesize':             Particlesize,
        'reactor_type':             reactor_type,
        'heatutility':              heatutility,
        'pressure_mode':            pressure_mode,
        'ExtractP_custom':          ExtractP_custom,
        'feed_dry_enable':          feed_dry_enable,
        'feed_dry_moisture':        feed_dry_moisture,
        'feed_dry_T':               feed_dry_T,
        'evap_n_effects':           evap_n_effects,
        'evap_target_solids':       evap_target_solids,
        'dryer_moisture':           dryer_moisture,
        'enable_recycle':           enable_recycle,
        'recycle_x_max':            recycle_x_max,
        'Yearstart':                Yearstart,
        'elec_price':               elec_price,
        'feed_price':               feed_price,
        'feed_storage_days':        feed_storage_days,
        'solv_storage_days':        solv_storage_days,
        'operator_hourly_wage':     operator_hourly_wage,
        'IRR':                      IRR,
        'duration':                 duration,
        'operating_days':           operating_days,
        'supervision_factor':       supervision_factor,
        'income_tax':               income_tax,
        'depreciation':             depreciation,
        'construction_schedule':    construction_schedule,
        'startup_months':           startup_months,
        'startup_FOCfrac':          startup_FOCfrac,
        'startup_VOCfrac':          startup_VOCfrac,
        'startup_salesfrac':        startup_salesfrac,
        'maintenance':              maintenance,
        'property_insurance':       property_insurance,
        'property_tax':             property_tax,
        'fringe_benefits':          fringe_benefits,
        'supplies':                 supplies,
        'administration':           administration,
        'Additional_direct_costs':  Additional_direct_costs,
        'indirect_costs':           indirect_costs,
        'WC_over_FCI':              WC_over_FCI,
        'finance_interest':         finance_interest,
        'finance_years':            finance_years,
        'finance_fraction':         finance_fraction,
    }


# ============================================================================
# Background parameter-sweep engine
# ----------------------------------------------------------------------------
# The sweep runs in a daemon thread that is fully decoupled from Streamlit's
# script-rerun cycle. It calls _run_one_simulation(..., display=False), which
# touches no Streamlit APIs, so widget interactions (fiddling the sidebar),
# websocket reconnects and tab refreshes no longer interrupt it. Each completed
# run is recorded both in a shared in-memory state object (for the live UI) and
# appended to a JSONL file on disk (so a reconnect can resume / re-download).
# The sweep only stops when every run is done OR the user presses Stop.
# ============================================================================

_SWEEP_DIR = os.path.join(tempfile.gettempdir(), "extraction_sweeps")
try:
    os.makedirs(_SWEEP_DIR, exist_ok=True)
except Exception:
    _SWEEP_DIR = tempfile.gettempdir()

# BioSTEAM uses a global flowsheet / settings, so two simulations must never
# run at the same time (background sweep worker + a Single Run, or two browser
# sessions sharing one Community Cloud process). This lock serialises them.
_SIM_LOCK = threading.Lock()


def _sweep_results_path(sweep_key):
    return os.path.join(_SWEEP_DIR, f"{sweep_key}.jsonl")


def _sweep_worker(state, base_params, combos, tea_mode_snapshot, results_path):
    """Execute the whole sweep in a background thread (no Streamlit calls)."""
    for i, overrides in enumerate(combos):
        # Honour an explicit Stop request (checked before each run).
        if state['stop'].is_set():
            break
        # Skip runs already completed before a reconnect/resume.
        if i < state['completed']:
            continue

        params = {**base_params, **overrides}
        if 'ExtractP_custom' in overrides:
            params['pressure_mode'] = 'Custom'

        _forced_foak = False
        if (tea_mode_snapshot == "Nth Plant"
                and params.get('reactor_type') == 'ultrasound'):
            params.update({k: v for k, v in TEA_DEFAULTS_FOAK.items()
                           if k in TEA_MODE_KEYS})
            params.update(overrides)  # explicit swept values still win
            _forced_foak = True

        run_start = time.time()
        try:
            with _SIM_LOCK:
                scalars = _run_one_simulation(params, display=False)
            run_dt = time.time() - run_start
            if scalars is not None:
                row = {**overrides, **scalars, 'tea_forced_foak': _forced_foak,
                       'run_time_s': run_dt, 'error': ''}
            else:
                row = {**overrides, 'tea_forced_foak': _forced_foak,
                       'run_time_s': run_dt, 'error': 'no result returned'}
            err_row = None
        except Exception as exc:
            run_dt = time.time() - run_start
            err_msg = f"{type(exc).__name__}: {exc}"
            row = {**overrides, 'tea_forced_foak': _forced_foak,
                   'run_time_s': run_dt, 'error': err_msg}
            err_row = {**overrides, 'error': err_msg}

        with state['lock']:
            state['results_rows'].append(row)
            if err_row is not None:
                state['error_rows'].append(err_row)
            state['times_per_run'].append(run_dt)
            state['completed'] = i + 1
            state['current_overrides'] = dict(overrides)
            state['last_update'] = time.time()

        # Durable, crash-safe append (survives a browser reconnect).
        try:
            with open(results_path, 'a') as fh:
                fh.write(json.dumps(row, default=str) + "\n")
        except Exception:
            pass

    with state['lock']:
        state['running'] = False
        state['finished_at'] = time.time()


def _start_sweep(sweep_values, base_params, combos, tea_mode_snapshot,
                 labels, fresh=False):
    """Build shared state, optionally resume from disk, and launch the thread.

    Returns the session_state key under which the sweep state is stored.
    """
    sweep_key = "sweep_" + hashlib.md5(
        str(sorted(sweep_values.items())).encode("utf-8")).hexdigest()[:16]
    results_path = _sweep_results_path(sweep_key)
    state_key = f"sweepstate_{sweep_key}"

    # If a sweep for this key is already running, stop it before relaunching.
    old = st.session_state.get(state_key)
    if old is not None and old.get('running') and old.get('stop') is not None:
        old['stop'].set()

    if fresh:
        try:
            if os.path.exists(results_path):
                os.remove(results_path)
        except Exception:
            pass

    # Resume: load any rows already on disk for this exact config.
    prior_rows, prior_errs, prior_times = [], [], []
    if os.path.exists(results_path):
        try:
            with open(results_path) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    prior_rows.append(r)
                    prior_times.append(float(r.get('run_time_s', 0.0) or 0.0))
                    if r.get('error'):
                        prior_errs.append(
                            {k: v for k, v in r.items()
                             if k not in ('tea_forced_foak', 'run_time_s')})
        except Exception:
            prior_rows, prior_errs, prior_times = [], [], []

    state = {
        'lock': threading.Lock(),
        'stop': threading.Event(),
        'results_rows': prior_rows,
        'error_rows': prior_errs,
        'times_per_run': prior_times,
        'completed': len(prior_rows),
        'current_overrides': {},
        'running': True,
        'started_at': time.time(),
        'last_update': time.time(),
        'finished_at': None,
        'n_total': len(combos),
        'sweep_values': dict(sweep_values),
        'labels': dict(labels),
        'results_path': results_path,
    }

    t = threading.Thread(
        target=_sweep_worker,
        args=(state, base_params, combos, tea_mode_snapshot, results_path),
        daemon=True,
    )
    t.start()
    state['thread'] = t
    st.session_state[state_key] = state
    st.session_state['sweep_active'] = state_key
    return state_key


def _render_sweep(state):
    """Render progress + (partial or final) results from the shared state.

    Returns True while the sweep is still running (so the caller knows whether
    to keep auto-refreshing).
    """
    with state['lock']:
        rows = list(state['results_rows'])
        err_rows = list(state['error_rows'])
        times = list(state['times_per_run'])
        completed = state['completed']
        running = state['running']
        started_at = state['started_at']
        finished_at = state['finished_at']
        current_overrides = dict(state.get('current_overrides') or {})
        n_total = state['n_total']
        labels = state['labels']
        sweep_values = state['sweep_values']
        results_path = state['results_path']

    # ---- Controls ----------------------------------------------------------
    if running:
        c1, c2 = st.columns([1, 3])
        with c1:
            if st.button("⏹️ Stop sweep", type="secondary",
                         use_container_width=True, key="sweep_stop_btn"):
                state['stop'].set()
                st.warning("Stopping after the current run finishes…")
        with c2:
            st.caption("The sweep runs in the background — you can change "
                       "sidebar values, switch tabs, or let the screen idle "
                       "without interrupting it. It stops only when you press "
                       "Stop or it finishes.")

    # ---- Progress ----------------------------------------------------------
    now = time.time()
    elapsed = (finished_at or now) - started_at
    frac = (completed / n_total) if n_total else 1.0
    st.progress(min(max(frac, 0.0), 1.0))

    if times:
        avg = sum(times) / len(times)
        if running:
            eta = avg * max(n_total - completed, 0)
            eta_text = (f"avg/run **{_fmt_duration(avg)}** · "
                        f"ETA **{_fmt_duration(eta)}**")
        else:
            eta_text = f"avg/run **{_fmt_duration(avg)}**"
    else:
        eta_text = "ETA: estimating after first run…"

    if running:
        ov_text = " · ".join(
            f"{labels.get(k, k)}="
            f"{(v if not isinstance(v, float) else f'{v:.4g}')}"
            for k, v in current_overrides.items()
        ) or "…"
        st.info(f"**Run {min(completed + 1, n_total)} of {n_total}** · "
                f"elapsed **{_fmt_duration(elapsed)}** · {eta_text}")
        st.caption(f"Current overrides: {ov_text}")
    else:
        n_ok = len([r for r in rows if not r.get('error')])
        n_fail = len(err_rows)
        if state['stop'].is_set() and completed < n_total:
            st.warning(f"⏹️ Sweep stopped — {completed} of {n_total} run(s) "
                       f"completed in **{_fmt_duration(elapsed)}** "
                       f"({n_ok} ok, {n_fail} failed). Press **Run Sweep** "
                       f"again to resume from here.")
        else:
            st.success(f"✅ Sweep complete — {completed} run(s) in "
                       f"**{_fmt_duration(elapsed)}** "
                       f"({n_ok} ok, {n_fail} failed)")

    if not rows:
        return running

    # ---- Results table -----------------------------------------------------
    df_results = pd.DataFrame(rows)
    if 'error' in df_results.columns:
        cols = [c for c in df_results.columns if c != 'error'] + ['error']
        df_results = df_results[cols]

    st.markdown("### 📋 Results"
                + ("  ·  *live*" if running else " preview"))
    st.dataframe(df_results, use_container_width=True, hide_index=True)

    # CSV download — works mid-sweep (downloads whatever is done so far).
    csv_bytes = df_results.to_csv(index=False).encode('utf-8')
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    st.download_button(
        label=("⬇️ Download results so far (CSV)" if running
               else "⬇️ Download results as CSV"),
        data=csv_bytes,
        file_name=f"extraction_sweep_{timestamp}.csv",
        mime="text/csv",
        type="primary",
        use_container_width=True,
        key="sweep_csv_dl",
    )

    if err_rows:
        with st.expander(f"⚠️ {len(err_rows)} run(s) failed — details",
                         expanded=False):
            st.dataframe(pd.DataFrame(err_rows),
                         use_container_width=True, hide_index=True)

    # ---- Single-parameter sensitivity plot (only once finished) ------------
    if not running:
        successful = (df_results[df_results['error'] == '']
                      if 'error' in df_results.columns else df_results)
        if (len(sweep_values) == 1 and len(successful) >= 2 and
                'product_price_USD_per_kg' in successful.columns):
            only_key = list(sweep_values.keys())[0]
            if only_key in successful.columns:
                try:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.plot(successful[only_key],
                            successful['product_price_USD_per_kg'],
                            marker='o')
                    ax.set_xlabel(labels.get(only_key, only_key))
                    ax.set_ylabel('Min. product price ($/kg)')
                    ax.set_title('Sensitivity of product price to '
                                 f'{labels.get(only_key, only_key)}')
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                except Exception:
                    pass

        # Offer a clean slate for the next run.
        if st.button("🗑️ Clear saved progress for this configuration",
                     key="sweep_clear_btn"):
            try:
                if os.path.exists(results_path):
                    os.remove(results_path)
            except Exception:
                pass
            st.session_state.pop('sweep_active', None)
            st.rerun()

    return running


if sim_mode == "Single Run" and run_simulation:
    with _SIM_LOCK:
        _run_one_simulation(_collect_base_params(), display=True)

elif sim_mode == "Parameter Sweep" and (run_sweep
                                        or st.session_state.get('sweep_active')):
    # ── Launch on button press (freezes ALL params at this instant, so any
    #    later sidebar fiddling cannot change an in-flight sweep) ──────────
    if run_sweep:
        _base_params = _collect_base_params()
        _combos = _build_combinations(sweep_values)
        _start_sweep(sweep_values, _base_params, _combos, tea_mode,
                     sweep_summary_labels)

    _active_key = st.session_state.get('sweep_active')
    _state = st.session_state.get(_active_key) if _active_key else None

    if _state is None:
        st.info("No active sweep. Tick the parameters to vary in the sidebar, "
                "set their ranges, then press **Run Sweep**.")
    else:
        st.subheader(f"\U0001F504 Parameter Sweep — {_state['n_total']} run(s)")
        _swept_list = ", ".join(
            f"`{_state['labels'].get(k, k)}`" for k in _state['sweep_values'])
        st.caption(f"Varying: {_swept_list}")

        # Live, decoupled rendering. With st.fragment only the progress panel
        # refreshes on a timer — the sidebar is never rebuilt by the refresh,
        # so it stays fully interactive while the sweep runs. On older
        # Streamlit without st.fragment we fall back to a whole-script poll.
        if hasattr(st, "fragment") and _state['running']:
            @st.fragment(run_every=2.0)
            def _live_sweep_panel():
                if not _render_sweep(_state):
                    st.rerun()  # finished → drop out to the static render path
            _live_sweep_panel()
        else:
            _still_running = _render_sweep(_state)
            if _still_running:
                time.sleep(1.5)
                st.rerun()

else:
    # Welcome message
    st.info(
        "👈 Configure the parameters in the sidebar. Pick **Single Run** "
        "for a full TEA report on one set of inputs, or **Parameter "
        "Sweep** to vary one or more inputs and export the results as "
        "a CSV."
    )

    st.markdown("""
    ### About This Tool

    This tool performs techno-economic analysis (TEA) of a solid-solvent extraction system. 

    **Features:**
    - Select from multiple feedstock types
    - Dry the feedstock before storage (rotary drum dryer, default 6 wt%
      moisture for stable long-term storage) so the stored pile is stable
      and feed water is not carried into the solvent loop or the product
    - Customize process conditions (temperature, flow rates, extraction time)
    - Choose reactor type (conventional, ultrasound-assisted, or microwave-assisted)
    - Adjust economic parameters (IRR, depreciation, operating costs, ect)
    - View detailed capital and operating cost breakdowns
    - Calculate minimum selling price for extract
    - **Parameter sweep mode**: vary one or more inputs over a range
      (linear or log spacing), run them all, and download the combined
      results as a CSV with live progress and ETA

    **Instructions:**
    1. Select a feedstock from the dropdown
    2. Configure process settings (feed flow, solvent, temperature, reactor type, etc.)
    3. Set the feedstock drying target (6% is the stock answer for stable
       long-term storage; pick "Custom" to enter your own)
    4. Adjust TEA settings (start year, IRR, depreciation method, etc.)
    5. *Single Run:* click "Run Simulation" to see the full report.
       *Parameter Sweep:* tick the parameters to vary in the sweep
       section, set their ranges and #points, then click "Run Sweep"
       and download the CSV when finished.

any questions or queries, contact KHeeley@ucdavis.edu, refer to EXBYCost in the subject
    """)

# Footer
st.markdown("---")
st.markdown("*Powered by BioSTEAM - Biological Systems Modeling Framework*")
