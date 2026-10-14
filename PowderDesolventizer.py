# -*- coding: utf-8 -*-
"""
PowderDesolventizer -- final solvent polishing of a dried extract powder.

An agitated vacuum contact dryer (conical screw "Nauta" type or horizontal
rotary vacuum paddle dryer) placed AFTER the product dryer (SD1) and BEFORE
the product cooler (SC1). It takes the powder from the product dryer's
residual solvent (e.g. 5 wt%) down to a ppm target (e.g. 25 ppm, the US limit
for hexane in spice oleoresins, 21 CFR 173.270).

    SD1 (spray / vacuum drum dryer) -> D401 (this unit) -> SC1 (cooler)
                                        |
                                        +-> solvent vapour -> condenser -> recycle

Model summary
-------------
Mass balance
    Only the solvent is removed. Everything else stays in the powder. The
    solvent left in the powder is set so that
        m_solvent_out / (m_other + m_solvent_out) = target_ppm * 1e-6.
    If the powder already meets the target the unit is a pass-through (no
    vessels, no duty, no cost).

Drying time (ASSUMPTION -- see the explanation document)
    Secondary drying of a spray-dried powder is a fast surface evaporation
    followed by slow Fickian diffusion of solvent out through the particle
    shell (Hsieh et al. 2015 [11]). Hsieh et al. model the shell as a slab
    (Crank 1975 [12]):
        M_t/M_0 = sum_n 8/((2n+1)^2 pi^2) * exp(-(2n+1)^2 * beta * t),
        beta = D pi^2 / (4 l^2)
    At long times only the n = 0 term matters, so
        M_t/M_0 ~ (8/pi^2) exp(-beta t)   ->   t ~ (1/beta) ln(M_0/M_t)
    i.e. drying time grows with the LOG of the reduction ratio. This is the
    classical first-order falling-rate result (Lewis 1921 [13]; Henderson &
    Pabis 1961 [14]).

      * time_basis='log_scaled' (default): a reference time t_ref is taken
        for 5 wt% -> 25 ppm and scaled for other start / target levels by
            t = t_ref * ln(X_in / X_target) / ln(X_ref_in / X_ref_target)
        (the constant ln(8/pi^2) = -0.21 is dropped; < 3 % against
        ln(2000) = 7.6 for the reference case).
        t_ref default 8.6 h, from Hsieh et al. 2015 Table IV: acetone in a
        spray-dried HPMCAS powder fell from 27,384 ppm to 23 ppm in 8 h in a
        tray dryer at 40 C. Rescaled to 5 wt% -> 25 ppm:
            8 h * ln(0.05/25e-6) / ln(27384/23) = 8 * 7.60 / 7.08 = 8.6 h.
        Conservative in two ways: the tool's jacket default is 60 C (Hsieh
        beta rises from 0.013 to 0.035 1/min from 40 to 60 C), and an agitated
        dryer was no slower than the tray dryer. Hsieh also found the drying
        rate independent of agitator tip speed (1 vs 2 m/s) and unchanged from
        25 g to 500 g, because the resistance is the particle shell, not the
        equipment -- so a lab time transfers to plant scale.
        NOT conservative for thick particles: time scales with l^2, so a
        drum-dried flake much thicker than a ~4 um spray-dried shell needs
        far longer. Hsieh also note the tail below M_t/M_0 ~ 0.15 is slower
        than the model; their measured times already include that.
      * time_basis='fixed': drying time = residence_time regardless of the
        start and target levels.

Vessel volume
    The vessels run semi-continuously (staggered fills, product drawn off as
    it finishes), so the powder hold-up is the feed rate times the drying
    time; no separate load / discharge time is added:
        V_working    = m_powder [kg/h] * t_drying [h] / bulk_density [kg/m3]
        V_total      = V_working / fill_fraction
        N_vessels    = max(n_min_vessels, ceil(V_total / max_vessel_volume))
        V_per_vessel = V_total / N_vessels

Energy
    Jacket duty = (latent heat of the solvent removed
                   + sensible heat to bring the powder up to T) / efficiency.
    Agitator power = installed drive rating per vessel from Perry's Table
    12-28, fitted to working volume (P = 4.45 * V_working**0.553 kW,
    R^2 = 0.85), times the number of vessels. An installed rating, so it
    overstates the running load. A fixed kW per m3 can be set instead with
    agitator_kW_per_m3.
    Vacuum system: BioSTEAM's VacuumSystem auxiliary (Seider et al. 2017),
    sized on total vessel volume and suction pressure.

Operating pressure
    P = pressure_ratio * Psat_solvent(T), capped at 1 atm. The default ratio
    is the same one the VacuumDrumDryer uses (ECN: 5 kPa / Psat_water(60 C)).

Purchase cost -- Perry's Chemical Engineers' Handbook, Table 12-28
    Size and cost data for batch rotary-agitator vacuum dryers. The table's
    (size, purchase cost) points are fitted to a power law in log-log space,
        C = a * V ** n
    and escalated to the simulation year with CEPCI:
        C_now = C * CEPCI_now / CEPCI_table.
    Cost is per vessel and multiplied by N_vessels. The bare-module factor
    (2.06) is the same Seider value the tool's other dryers use.

    The 8th-edition table (1995 prices, stainless steel 304) is entered in
    PERRY_12_28 below. The unit raises a clear error if the table is
    emptied, so it can never silently produce a zero-cost dryer.

References
----------
[1] Green, D. W.; Perry, R. H. (eds.) Perry's Chemical Engineers'
    Handbook, 8th ed.; McGraw-Hill, 2008. Sec. 12, Psychrometry, Evaporative
    Cooling, and Solids Drying: Table 12-28 (size and cost data for rotary
    agitator vacuum dryers, Stokes Vacuum Inc., p. 12-72): purchase price,
    working capacity and drive power.
[2] Schluender, E. U.; Mollekopf, N. Vacuum contact drying of free flowing
    mechanically agitated particulate material. Drying '85 (1985).
    doi:10.1007/978-3-662-21830-3_8
[3] Tsotsas, E.; Kwapinska, M.; Saage, G. Modeling of contact dryers.
    Drying Technology 2007, 25 (7-8), 1377-1391.
[4] Hoekstra, L.; Vonk, P.; Hulshof, L. A. Modeling the scale-up of contact
    drying processes. Org. Process Res. Dev. 2006, 10 (3), 409-416.
[5] Murru, M.; Giorgio, G.; Montomoli, S.; Ricard, F.; Stepanek, F.
    Model-based scale-up of vacuum contact drying of pharmaceutical
    compounds. Chem. Eng. Sci. 2011.
[6] Yazdanpanah, N. Predictive drying recipe development for drying a
    heat-sensitive and sticky powder. AIChE Annual Meeting 2024, 725g
    (conference abstract; context only).
[7] US Patent 9,223,238. Production of latex using a wipe film evaporator
    (context only).
[8] Seider, W. D. et al. Product and Process Design Principles, 4th ed.;
    Wiley, 2017 (bare-module factor; vacuum system costing in BioSTEAM).
[9] Hosokawa Micron, Nauta vacuum/pressure dryer (U = 50-2000 W/m2/K;
    sizes > 20,000 L); S2 Engineering rotary vacuum paddle dryers
    (500-10,000 L); Buflovak vacuum rotary paddle dryers (working capacity
    quoted at 60 % full).
[10] De Smidt, R. P.; Wemmers, A. K.; Spoelstra, S. Thin film drying
    processes. ECN-E--17-084, 2017 (vacuum pressure ratio).
[11] Hsieh, D. S.; Yue, H.; Nicholson, S. J.; Roberts, D.; Schild, R.;
    Gamble, J. F.; Lindrud, M. The secondary drying and the fate of organic
    solvents for spray dried dispersion drug product. Pharm. Res. 2015, 32,
    1804-1816. doi:10.1007/s11095-014-1577-y
    (Eqs. 2, 4, 5; Tables II-IV; Figs. 12-15.)
[12] Crank, J. The Mathematics of Diffusion, 2nd ed.; Clarendon Press:
    Oxford, 1975 (plane sheet series solution).
[13] Lewis, W. K. The rate of drying of solid materials. Ind. Eng. Chem.
    1921, 13 (5), 427-432.
[14] Henderson, S. M.; Pabis, S. Grain drying theory I: Temperature effect
    on drying coefficient. J. Agric. Eng. Res. 1961, 6, 169-174.
"""
from math import ceil, exp, log

import biosteam as bst
import thermosteam as tmo

__all__ = ('PowderDesolventizer', 'perry_cost_data_ready',
           'PERRY_12_28', 'DESOLV_DEFAULTS')


# ============================================================================
# Perry's Table 12-28 cost data -- ENTER FROM YOUR COPY OF PERRY'S
# ----------------------------------------------------------------------------
# 'points'     : list of (size, purchase_cost_USD) rows from the table.
#                Two or more rows are needed for the power-law fit.
# 'size_basis' : 'working' if the table's size column is working capacity,
#                'total' if it is total vessel volume.
# 'size_unit'  : 'm3', 'L', 'ft3' or 'gal' -- the unit of the size column as
#                printed; it is converted to m3 here.
# 'CEPCI'      : Chemical Engineering Plant Cost Index for the year the
#                table's prices refer to (stated in the table footnote).
# 'edition'    : free text for traceability, e.g. "Perry's 8th ed. (2008)".
# ============================================================================
PERRY_12_28 = {
    # Perry's Chemical Engineers' Handbook, 8th ed. (2008), p. 12-72,
    # Table 12-28 "Standard Rotary Vacuum Dryers" (Stokes Vacuum, Inc.).
    # Purchase price (1995), STAINLESS STEEL (304) column -- food contact.
    # Price includes shell, 50 psig jacket, agitator, drive and motor.
    # Working capacity = "loading with product level on or around the
    # agitator shaft". Power-law fit over these 7 rows:
    #     C = 203,234 * V_working[m3] ** 0.387   (R^2 = 0.986, all rows
    #     within +/-14 %); valid 0.028-8.33 m3 working per vessel.
    'edition': "Perry's 8th ed. (2008), Table 12-28, stainless steel (304)",
    'CEPCI': 381.1,         # 1995 annual composite CEPCI (Chemical
                            # Engineering, CEPCI_2002.pdf, Table 2)
    'size_basis': 'working',
    'size_unit': 'm3',
    'points': [
        # (working capacity m3, purchase price 1995 USD, SS 304)
        (0.028,  53000),
        (0.283, 130000),
        (0.991, 180000),
        (1.42,  205000),
        (3.57,  380000),
        (6.94,  440000),
        (8.33,  465000),
    ],
}

_SIZE_TO_M3 = {'m3': 1.0, 'l': 1e-3, 'ft3': 0.0283168, 'gal': 0.00378541}


def perry_cost_data_ready(table=None):
    """True if the Perry's table has enough data to cost the unit."""
    t = PERRY_12_28 if table is None else table
    pts = [p for p in t.get('points', ()) if p and p[0] > 0 and p[1] > 0]
    return (len(pts) >= 2 and t.get('CEPCI')
            and t.get('size_unit', '').lower() in _SIZE_TO_M3
            and t.get('size_basis') in ('working', 'total'))


def _fit_perry(table):
    """Least-squares power law C = a * V**n on the table points (V in m3).

    Returns (a, n, V_min, V_max) with V in m3 on the table's size basis.
    """
    if not perry_cost_data_ready(table):
        raise RuntimeError(
            "PowderDesolventizer: Perry's Table 12-28 cost data has not been "
            "entered. Fill PERRY_12_28 in PowderDesolventizer.py with at "
            "least two (size, purchase cost) rows, the size unit and basis, "
            "and the CEPCI of the table's cost year.")
    f = _SIZE_TO_M3[table['size_unit'].lower()]
    pts = [(v * f, c) for v, c in table['points'] if v > 0 and c > 0]
    xs = [log(v) for v, _ in pts]
    ys = [log(c) for _, c in pts]
    k = len(xs)
    xm, ym = sum(xs) / k, sum(ys) / k
    sxx = sum((x - xm) ** 2 for x in xs)
    if sxx <= 0:
        raise RuntimeError("PowderDesolventizer: Perry's table points must "
                           "span more than one size.")
    n = sum((x - xm) * (y - ym) for x, y in zip(xs, ys)) / sxx
    a = exp(ym - n * xm)
    V = [v for v, _ in pts]
    return a, n, min(V), max(V)


# Default assumptions (also used by the Streamlit sidebar). Every value here
# is an ASSUMPTION or a typical value; see the explanation document.
DESOLV_DEFAULTS = {
    'target_ppm': 25.0,          # 21 CFR 173.270 (hexane, spice oleoresins)
    'residence_time': 8.6,       # h, 5 wt% -> 25 ppm; Hsieh et al. 2015
                                 # Table IV rescaled (see module docstring)
    'ref_X_in': 0.05,            # reference start (5 wt%, SD1 default)
    'ref_ppm': 25.0,             # reference target
    'bulk_density': 450.0,       # kg/m3, spray-dried food powder (300-600)
    'fill_fraction': 0.6,        # working / total volume (Buflovak, 60 % full)
    'T_C': 60.0,                 # jacket / product temperature, C
    'n_min_vessels': 2,          # staggered batches for a continuous plant
    'max_vessel_volume': 10.0,   # m3 total; S2 paddle dryers to 10,000 L
    'thermal_efficiency': 0.95,  # jacket heat loss (same as VacuumDrumDryer)
    'agitator_kW_per_m3': None,  # None -> Perry's 12-28 drive-power fit
}


class PowderDesolventizer(bst.Unit):
    """Agitated vacuum contact dryer polishing a powder to a ppm solvent target.

    ins  : [0] dried powder from the product dryer
    outs : [0] solvent vapour, [1] polished powder

    See the module docstring for the model and its sources.
    """
    _N_ins = 1
    _N_outs = 2
    auxiliary_unit_names = ('vacuum_system',)
    _F_BM_default = {'Desolventizer vessels': 2.06}
    _units = {
        'Drying time': 'hr',
        'Working volume': 'm3',
        'Total volume': 'm3',
        'Volume per vessel': 'm3',
        'Number of vessels': '',
        'Solvent removed': 'kg/hr',
        'Inlet solvent': 'ppm',
        'Outlet solvent': 'ppm',
        'Pressure': 'kPa',
        'Agitator power': 'kW',
    }

    def _init(self, solvent='Water',
              target_ppm=DESOLV_DEFAULTS['target_ppm'],
              residence_time=DESOLV_DEFAULTS['residence_time'],
              time_basis='log_scaled',
              ref_X_in=DESOLV_DEFAULTS['ref_X_in'],
              ref_ppm=DESOLV_DEFAULTS['ref_ppm'],
              bulk_density=DESOLV_DEFAULTS['bulk_density'],
              fill_fraction=DESOLV_DEFAULTS['fill_fraction'],
              T=DESOLV_DEFAULTS['T_C'] + 273.15,
              pressure_ratio=None,
              n_min_vessels=DESOLV_DEFAULTS['n_min_vessels'],
              max_vessel_volume=DESOLV_DEFAULTS['max_vessel_volume'],
              thermal_efficiency=DESOLV_DEFAULTS['thermal_efficiency'],
              agitator_kW_per_m3=DESOLV_DEFAULTS['agitator_kW_per_m3'],
              vacuum_system_preference='Liquid-ring pump',
              perry_table=None):
        self.solvent = solvent
        self.target_ppm = target_ppm
        self.residence_time = residence_time
        if time_basis not in ('log_scaled', 'fixed'):
            raise ValueError("time_basis must be 'log_scaled' or 'fixed'")
        self.time_basis = time_basis
        self.ref_X_in = ref_X_in
        self.ref_ppm = ref_ppm
        self.bulk_density = bulk_density
        self.fill_fraction = fill_fraction
        self.T = T
        if pressure_ratio is None:
            # ECN vacuum drum case: 5 kPa at 60 C product -> 0.251 for water.
            pressure_ratio = 5000. / tmo.Chemical('Water').Psat(333.15)
        self.pressure_ratio = pressure_ratio
        self.n_min_vessels = int(n_min_vessels)
        self.max_vessel_volume = max_vessel_volume
        self.thermal_efficiency = thermal_efficiency
        self.agitator_kW_per_m3 = agitator_kW_per_m3
        self.vacuum_system_preference = vacuum_system_preference
        self.perry_table = perry_table   # None -> module-level PERRY_12_28
        self.vacuum_system = None
        self.P = 101325.
        self.X_in = 0.

    # -- mass balance --------------------------------------------------------
    def _run(self):
        feed = self.ins[0]
        vapor, product = self.outs
        solv = self.solvent
        product.copy_like(feed)
        vapor.empty()
        vapor.phase = 'g'
        try:
            Psat = self.chemicals[solv].Psat(self.T)
            self.P = min(self.pressure_ratio * Psat, 101325.)
        except Exception:
            self.P = 101325.

        m_tot = product.F_mass
        m_solv = product.imass[solv]
        self.X_in = m_solv / m_tot if m_tot > 0 else 0.
        x_t = self.target_ppm * 1e-6
        if m_tot > 0 and self.X_in > x_t:
            m_other = m_tot - m_solv
            m_keep = x_t * m_other / (1. - x_t)
            vapor.imass[solv] = m_solv - m_keep
            product.imass[solv] = m_keep
        vapor.T = product.T = self.T
        vapor.P = product.P = self.P

    # -- drying time ---------------------------------------------------------
    def _drying_time(self):
        x_t = self.target_ppm * 1e-6
        if self.X_in <= x_t:
            return 0.
        if self.time_basis == 'fixed':
            return self.residence_time
        ref = log(self.ref_X_in / (self.ref_ppm * 1e-6))
        return self.residence_time * log(self.X_in / x_t) / ref

    # -- sizing and energy ---------------------------------------------------
    def _design(self):
        D = self.design_results
        feed = self.ins[0]
        vapor, product = self.outs
        t_dry = self._drying_time()
        self.vacuum_system = None
        D['Inlet solvent'] = self.X_in * 1e6
        D['Outlet solvent'] = (product.imass[self.solvent] / product.F_mass
                               * 1e6 if product.F_mass > 0 else 0.)
        D['Solvent removed'] = vapor.F_mass
        D['Pressure'] = self.P / 1e3
        if t_dry <= 0 or feed.F_mass <= 0:
            # Pass-through: already at or below the target.
            for k in ('Drying time', 'Working volume',
                      'Total volume', 'Volume per vessel', 'Number of vessels',
                      'Agitator power'):
                D[k] = 0.
            return

        # Semi-continuous: hold-up = feed rate x drying time.
        V_work = feed.F_mass * t_dry / self.bulk_density
        V_tot = V_work / self.fill_fraction
        N = max(self.n_min_vessels, ceil(V_tot / self.max_vessel_volume))
        D['Drying time'] = t_dry
        D['Working volume'] = V_work
        D['Total volume'] = V_tot
        D['Volume per vessel'] = V_tot / N
        D['Number of vessels'] = N
        if self.agitator_kW_per_m3 is None:
            # Perry's Table 12-28 drive rating vs working capacity
            # (fit: P = 4.45 * V**0.553 kW per vessel).
            V_work_each = V_work / N
            D['Agitator power'] = N * 4.447 * V_work_each ** 0.553
        else:
            D['Agitator power'] = self.agitator_kW_per_m3 * V_work

        # Jacket duty: latent heat of the solvent removed + sensible heat to
        # bring the powder to T (a hotter powder is not credited).
        chem = self.chemicals[self.solvent]
        T = self.T
        try:
            Hvap = chem.Hvap(min(T, chem.Tb))          # J/mol
        except Exception:
            Hvap = chem.Hvap(chem.Tb)
        latent = vapor.F_mol * Hvap                    # kmol/h * J/mol = kJ/h
        sensible = feed.C * max(0., T - feed.T)        # kJ/h
        duty = (latent + sensible) / self.thermal_efficiency
        if duty > 0:
            # BioSTEAM rejects heating with T_in > T_out; a powder arriving
            # hotter than the jacket setpoint is simply held at T.
            self.add_heat_utility(duty, T_in=min(feed.T, T - 0.1), T_out=T)

        if self.P < 101325.:
            self.vacuum_system = bst.VacuumSystem(
                self, self.vacuum_system_preference,
                P_suction=self.P, vessel_volume=V_tot)

    # -- cost ----------------------------------------------------------------
    def _cost(self):
        D = self.design_results
        C = self.baseline_purchase_costs
        N = D.get('Number of vessels', 0)
        if not N:
            C['Desolventizer vessels'] = 0.
            return
        table = self.perry_table or PERRY_12_28
        a, n, V_lo, V_hi = _fit_perry(table)
        V_each = D['Volume per vessel']
        if table['size_basis'] == 'working':
            V_each *= self.fill_fraction
        self.perry_fit = {'a': a, 'n': n, 'V_min_m3': V_lo, 'V_max_m3': V_hi,
                          'V_used_m3': V_each,
                          'in_range': V_lo <= V_each <= V_hi}
        C['Desolventizer vessels'] = (N * a * V_each ** n
                                      * bst.CE / table['CEPCI'])
        self.add_power_utility(D['Agitator power'])
