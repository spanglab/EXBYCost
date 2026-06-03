"""
Solid-Solvent Extraction Unit for BioSTEAM

This module implements a continuous solid-solvent extractor unit operation
that performs combined solid-liquid and liquid-liquid equilibrium separations,
with extraction kinetics based on the Reddy-Doraiswamy diffusion model.

The equilibrium calculations determine maximum extractable amounts, while the
kinetic model determines how much is actually extracted given the residence time.

"""

import biosteam as bst
from biosteam.units.decorators import cost
from biosteam import Unit
import thermosteam as tmo
from thermosteam import Stream
import numpy as np
from math import pi

# Import HSP-based solubility prediction for compounds without UNIFAC groups
try:
    from Polymersolubility import (
        predict_solubility,
        predict_compound_solubility,
        POLYMER_DB,
        SOLVENT_DB
    )
    HSP_SOLUBILITY_AVAILABLE = True
except ImportError:
    HSP_SOLUBILITY_AVAILABLE = False


# =============================================================================
#                    REDDY-DORAISWAMY KINETIC MODEL FUNCTIONS
# =============================================================================

def calculate_Dab_RD(T, viscosity, Molvol_solute_Tb, Molvol_solvent_Tb, molmassS):
    """
    Calculate diffusion coefficient using Reddy-Doraiswamy (RD) correlation.

    Dab = K * (Ms^0.5 * T) / (mu * (VA*VB)^0.333)

    Where:
        K = 8.5e-8 if VA/VB < 1.5, else 1.0e-7
        VB = molar volume of solvent at its normal boiling point (cm^3/mol)
        VA = molar volume of solute at its normal boiling point (cm^3/mol)
        T = temperature (K)
        mu = viscosity (cP)
        Ms = molecular weight of solvent (g/mol)

    Parameters
    ----------
    T : float
        Temperature in Kelvin
    viscosity : float
        Solvent viscosity in cP (centipoise)
    Molvol_solute_Tb : float
        Molar volume of solute at its normal boiling point (cm³/mol)
    Molvol_solvent_Tb : float
        Molar volume of solvent at its normal boiling point (cm³/mol)
    molmassS : float
        Molecular weight of solvent (g/mol)

    Returns
    -------
    Dab : float
        Diffusion coefficient in cm²/s
    K : float
        Correlation constant used
    VA_VB : float
        Molar volume ratio
    """
    VA_VB = Molvol_solute_Tb / Molvol_solvent_Tb
    K = 8.5e-8 if VA_VB < 1.5 else 1.0e-7

    Dab = (K * (molmassS**0.5) * T) / (viscosity * (Molvol_solvent_Tb * Molvol_solute_Tb)**0.33333)
    return Dab, K, VA_VB


def calculate_k1_from_Dab(Dab, ParticleR, X, Cs):
    """
    Calculate the rate constant k1 from diffusion coefficient.

    k1 = R² / (15 * π * Dab * X * Cs)

    Parameters
    ----------
    Dab : float
        Diffusion coefficient (cm²/s)
    ParticleR : float
        Particle radius (cm)
    X : float
        Porosity/tortuosity ratio (dimensionless)
    Cs : float
        Initial/reference concentration

    Returns
    -------
    k1 : float
        Rate constant for extraction kinetics
    """
    if Cs <= 0 or Dab <= 0:
        return float('inf')
    k1 = (ParticleR**2) / (15 * pi * Dab * X * Cs)
    return k1


def extraction_kinetics(t, k1, EQcomp):
    """
    Extraction kinetics model based on diffusion.
    Returns concentration as a function of time.

    C(t) = t / (k1 + t/EQcomp)

    As t -> infinity, C -> EQcomp

    Parameters
    ----------
    t : float or array
        Time in seconds
    k1 : float
        Rate constant
    EQcomp : float
        Equilibrium concentration (maximum extractable)

    Returns
    -------
    C : float or array
        Concentration at time t
    """
    if k1 == float('inf') or EQcomp <= 0:
        return 0.0
    if isinstance(t, np.ndarray):
        return np.where(t > 0, t / (k1 + t / EQcomp), 0)
    else:
        return t / (k1 + t / EQcomp) if t > 0 else 0.0


def calculate_extraction_fraction(t, k1, EQcomp):
    """
    Calculate the fraction of equilibrium achieved at time t.

    Parameters
    ----------
    t : float
        Time in seconds
    k1 : float
        Rate constant
    EQcomp : float
        Equilibrium concentration

    Returns
    -------
    fraction : float
        Fraction of equilibrium achieved (0 to 1)
    """
    if EQcomp <= 0:
        return 1.0
    C_t = extraction_kinetics(t, k1, EQcomp)
    return min(C_t / EQcomp, 1.0)


# =============================================================================
#                    SOLID-LIQUID EQUILIBRIUM FUNCTION
# =============================================================================

def has_unifac_groups(chemical) -> bool:
    """
    Check if a chemical has valid UNIFAC groups for activity coefficient calculations.

    Parameters
    ----------
    chemical : thermosteam.Chemical
        The chemical object to check

    Returns
    -------
    bool
        True if chemical has usable UNIFAC groups, False otherwise
    """
    # Check for Dortmund UNIFAC groups (preferred for activity coefficients)
    if hasattr(chemical, 'Dortmund'):
        groups = chemical.Dortmund
        if groups is not None and len(groups) > 0:
            return True

    # Check for standard UNIFAC groups
    if hasattr(chemical, 'UNIFAC'):
        groups = chemical.UNIFAC
        if groups is not None and len(groups) > 0:
            return True

    # Legacy attribute names (for compatibility)
    if hasattr(chemical, 'UNIFAC_Dortmund_groups'):
        groups = chemical.UNIFAC_Dortmund_groups
        if groups is not None and len(groups) > 0:
            return True

    if hasattr(chemical, 'UNIFAC_groups'):
        groups = chemical.UNIFAC_groups
        if groups is not None and len(groups) > 0:
            return True

    return False


def is_polymer(compound_name: str) -> bool:
    """
    Determine if a compound is likely a polymer based on naming conventions.

    Parameters
    ----------
    compound_name : str
        Name of the compound

    Returns
    -------
    bool
        True if compound appears to be a polymer
    """
    name_lower = compound_name.lower().strip()

    # Check if in polymer database
    if HSP_SOLUBILITY_AVAILABLE and name_lower in POLYMER_DB:
        return True

    # Check for "poly" at the START of the name (not middle like "monopoly")
    if name_lower.startswith('poly'):
        return True

    # Check common polymer names that don't start with "poly"
    polymer_names = [
        'cellulose', 'lignin', 'hemicellulose', 'starch',
        'chitin', 'chitosan', 'pectin', 'xylan', 'mannan',
        'rubber', 'nylon', 'teflon', 'pvc', 'pet', 'hdpe', 'ldpe',
        'pmma', 'abs', 'pla', 'pha', 'phb'
    ]

    for polymer_name in polymer_names:
        if polymer_name in name_lower:
            return True

    return False


def calculate_hsp_solubility_extraction(compound_name: str, solvent_name: str,
                                        compound_feed_mol: float, solvent_mol: float,
                                        temperature: float, chemical,
                                        pressure: float = 101325) -> dict:
    """
    Calculate extraction using Hansen Solubility Parameter method.

    For compounds without UNIFAC groups, uses Flory-Huggins based prediction
    from the Polymersolubility module.

    Parameters
    ----------
    compound_name : str
        Name of the compound to extract
    solvent_name : str
        Name of the solvent
    compound_feed_mol : float
        Molar feed rate of compound (mol/hr)
    solvent_mol : float
        Molar flow of solvent (mol/hr)
    temperature : float
        Temperature in Kelvin
    chemical : thermosteam.Chemical
        Chemical object for molecular weight lookup
    pressure : float, optional
        Operating pressure in Pa. Default is 101325 Pa.

    Returns
    -------
    dict with:
        - dissolved_mol: Amount dissolved in solvent (mol)
        - solid_mol: Amount remaining as solid (mol)
        - solubility_g_L: Predicted solubility (g/L)
        - method: 'polymer' or 'compound'
        - success: bool
        - error: str if failed
    """
    if not HSP_SOLUBILITY_AVAILABLE:
        return {
            'dissolved_mol': compound_feed_mol,  # Assume fully soluble as fallback
            'solid_mol': 0.0,
            'solubility_g_L': None,
            'method': 'fallback_assumed_soluble',
            'success': False,
            'error': 'HSP solubility module not available'
        }

    # Normalize solvent name for lookup
    solvent_lookup = solvent_name.lower().strip()

    # Check if solvent is in HSP database
    if solvent_lookup not in SOLVENT_DB:
        # Try common aliases
        solvent_aliases = {
            'etoh': 'ethanol',
            'meoh': 'methanol',
            'h2o': 'water',
            'dcm': 'dichloromethane',
            'dmf': 'dmf',
            'thf': 'thf',
            'mecn': 'acetonitrile',
            'etac': 'ethyl acetate',
        }
        solvent_lookup = solvent_aliases.get(solvent_lookup, solvent_lookup)

        if solvent_lookup not in SOLVENT_DB:
            return {
                'dissolved_mol': compound_feed_mol,
                'solid_mol': 0.0,
                'solubility_g_L': None,
                'method': 'fallback_solvent_not_in_db',
                'success': False,
                'error': f'Solvent {solvent_name} not in HSP database'
            }

    try:
        # Determine if polymer or small molecule and call appropriate function
        if is_polymer(compound_name):
            result = predict_solubility(compound_name, solvent_lookup, temperature)
            method = 'polymer'
        else:
            result = predict_compound_solubility(compound_name, solvent_lookup, temperature)
            method = 'compound'

        if 'error' in result:
            return {
                'dissolved_mol': compound_feed_mol,
                'solid_mol': 0.0,
                'solubility_g_L': None,
                'method': f'fallback_{method}_error',
                'success': False,
                'error': result['error']
            }

        solubility_g_L = result['solubility_g_L']

        # Get molecular weight
        MW = chemical.MW if hasattr(chemical, 'MW') else 100.0  # Default if not available

        # Convert solubility to mol/L
        solubility_mol_L = solubility_g_L / MW

        # Estimate solvent volume (L) from molar flow
        # Get solvent molar volume at operating temperature and pressure
        solvent_chem = None
        try:
            solvent_chem = tmo.settings.get_thermo().chemicals[solvent_name]
            Vm_solvent = solvent_chem.V.l(temperature, pressure)  # m³/mol
            solvent_volume_L = solvent_mol * Vm_solvent * 1000  # Convert to L
        except:
            # Fallback: estimate from typical molar volumes
            typical_Vm = {
                'water': 0.018, 'ethanol': 0.0585, 'methanol': 0.0407,
                'hexane': 0.1316, 'toluene': 0.1068, 'acetone': 0.074,
                'dmso': 0.0713, 'chloroform': 0.0807
            }
            Vm_L = typical_Vm.get(solvent_lookup, 0.1)  # Default 100 mL/mol
            solvent_volume_L = solvent_mol * Vm_L

        # Calculate maximum dissolvable amount
        max_dissolved_mol = solubility_mol_L * solvent_volume_L

        # Actual dissolved is minimum of feed and solubility limit
        dissolved_mol = min(compound_feed_mol, max_dissolved_mol)
        solid_mol = compound_feed_mol - dissolved_mol

        return {
            'dissolved_mol': dissolved_mol,
            'solid_mol': solid_mol,
            'solubility_g_L': solubility_g_L,
            'solubility_mol_L': solubility_mol_L,
            'method': method,
            'success': True,
            'error': None,
            'chi': result.get('chi', None),
            'Ra': result.get('Ra', None)
        }

    except Exception as e:
        return {
            'dissolved_mol': compound_feed_mol,
            'solid_mol': 0.0,
            'solubility_g_L': None,
            'method': 'fallback_exception',
            'success': False,
            'error': str(e)
        }



def calculate_multicomponent_sle_iterative(solute_feeds, solvent, solvent_flow,
                                           temperature, max_iter=20, tol=0.001):
    """
    Calculate multicomponent SLE with proper iteration to avoid local minima.
    All solutes compete for dissolution in the solvent.
    """
    solutes = list(solute_feeds.keys())
    total_amounts = solute_feeds.copy()
    liquid_amounts = total_amounts.copy()

    for iteration in range(max_iter):
        old_liquid_amounts = liquid_amounts.copy()
        new_liquid_amounts = {}

        for solute in solutes:
            try:
                mixture_components = []
                mixture_components.append((solute, total_amounts[solute]))

                for other_solute in solutes:
                    if other_solute != solute:
                        mixture_components.append((other_solute, liquid_amounts[other_solute]))

                mixture_components.append((solvent, solvent_flow))

                imol = tmo.indexer.MolarFlowIndexer(l=mixture_components, phases=('s', 'l'))
                sle = tmo.equilibrium.SLE(imol)
                sle(solute, T=temperature)

                solute_index = sle.chemicals.get_index(solute)
                liquid_mol = sle._liquid_mol[solute_index]

                new_liquid_amounts[solute] = min(liquid_mol, total_amounts[solute])

            except Exception:
                new_liquid_amounts[solute] = total_amounts[solute]

        liquid_amounts = new_liquid_amounts.copy()

        max_change = max([abs(liquid_amounts[s] - old_liquid_amounts[s]) / total_amounts[s]
                          for s in solutes if total_amounts[s] > 0])

        if max_change < tol:
            solid_amounts = {s: total_amounts[s] - liquid_amounts[s] for s in solutes}
            return liquid_amounts, solid_amounts, iteration + 1

    solid_amounts = {s: total_amounts[s] - liquid_amounts[s] for s in solutes}
    return liquid_amounts, solid_amounts, max_iter


# =============================================================================
#                    LIQUID-LIQUID EQUILIBRIUM FUNCTIONS
# =============================================================================

def solve_lle_with_consensus(scaled_feed_dict, scaled_excess_amount, target_compound,
                             temperature, solvent, num_attempts=3, max_deviation=3.0):
    """
    Solve LLE with multiple initial conditions and check for consensus.
    """
    perturbations = [1.0, 1.05, 0.95, 1.1, 0.9]
    results = []

    for i, perturb in enumerate(perturbations[:num_attempts]):
        try:
            if i == 0:
                imol_excess = tmo.indexer.MolarFlowIndexer(
                    l=[(k, v) for k, v in scaled_feed_dict.items()],
                    L=[(target_compound, scaled_excess_amount)]
                )
            else:
                imol_excess = tmo.indexer.MolarFlowIndexer(
                    l=[(k, v * perturb) for k, v in scaled_feed_dict.items()],
                    L=[(target_compound, scaled_excess_amount * (2.0 - perturb))]
                )

            lle = tmo.equilibrium.LLE(imol_excess)
            lle(T=temperature)

            chemicals_list = list(lle.chemicals.IDs)
            total_mol_l = sum(lle.imol['l', chem] for chem in chemicals_list)
            total_mol_L = sum(lle.imol['L', chem] for chem in chemicals_list)

            if total_mol_l < 1e-10 or total_mol_L < 1e-10:
                continue

            solvent_in_l = lle.imol['l', solvent]
            solvent_in_L = lle.imol['L', solvent]

            if solvent_in_l > solvent_in_L:
                extract_phase = 'l'
            else:
                extract_phase = 'L'

            dissolved = lle.imol[extract_phase, target_compound]
            solvent_in_extract = lle.imol[extract_phase, solvent]

            if solvent_in_extract > 1e-10:
                solubility_ratio = dissolved / solvent_in_extract
                results.append({
                    'lle': lle,
                    'solubility_ratio': solubility_ratio,
                    'extract_phase': extract_phase
                })
        except Exception:
            continue

    if len(results) == 0:
        raise RuntimeError("All LLE consensus attempts failed")

    solubilities = [r['solubility_ratio'] for r in results]

    is_reliable = True
    if len(results) > 1:
        max_sol = max(solubilities)
        min_sol = min(solubilities)
        max_ratio = max_sol / min_sol if min_sol > 1e-10 else float('inf')
        is_reliable = max_ratio <= max_deviation

        if not is_reliable:
            median_idx = np.argsort(solubilities)[len(solubilities) // 2]
            best_result = results[median_idx]
        else:
            best_result = results[0]
    else:
        best_result = results[0]

    return best_result['lle'], is_reliable, best_result['solubility_ratio']


def sequential_lle_separation(feed, temperature, solvent, thermo=None, verbose=False):
    """
    Performs sequential LLE separation on liquid compounds in a feed.
    """
    if verbose:
        print("=" * 60)
        print("Sequential LLE Separation")
        print("=" * 60)

    if thermo is None:
        thermo = tmo.settings.get_thermo()

    chemicals = [chem.ID for chem in thermo.chemicals]
    separation_status = {}

    liquid_compounds = {}
    for compound_name, amount in feed.items():
        if compound_name == solvent:
            continue

        chem = thermo.chemicals[compound_name]
        Tm = chem.Tm
        Tb = chem.Tb

        if Tm is not None and Tb is not None:
            if Tm < temperature < Tb:
                liquid_compounds[compound_name] = amount
                if verbose:
                    print(f"{compound_name}: Liquid at {temperature}K (Tm={Tm:.1f}K, Tb={Tb:.1f}K) - {amount} mol")
            else:
                if verbose:
                    print(f"{compound_name}: Not liquid at {temperature}K (Tm={Tm:.1f}K, Tb={Tb:.1f}K)")
                separation_status[compound_name] = 'Not_Liquid'
        else:
            if verbose:
                print(f"{compound_name}: Missing phase data, skipping")
            separation_status[compound_name] = 'Missing_Data'

    if not liquid_compounds:
        if verbose:
            print("\nNo liquid compounds found in feed (excluding solvent)!")
        extract_composition = feed.copy()
        residual_composition = {}
        return extract_composition, residual_composition, separation_status

    if verbose:
        print(f"\nFound {len(liquid_compounds)} liquid compound(s) to process")

    sorted_liquid_compounds = sorted(liquid_compounds.items(), key=lambda x: x[1], reverse=True)
    current_feed_state = feed.copy()
    non_liquid_compounds = {k: v for k, v in feed.items()
                           if k not in liquid_compounds and k != solvent}
    total_residual_amounts = {}

    for idx, (target_compound, actual_compound_amount) in enumerate(sorted_liquid_compounds, 1):

        if verbose:
            print("\n" + "=" * 60)
            print(f"Step {idx}/{len(sorted_liquid_compounds)}: Processing {target_compound}")
            print("=" * 60)

        base_solvent_amount = current_feed_state[solvent]
        excess_compound = base_solvent_amount

        phase_separation_achieved = False
        scaling_factor = 1.0
        max_scaling_attempts = 25
        scaling_multiplier = 1.5

        final_lle_excess = None
        final_solubility_ratio = None
        final_is_reliable = None

        for attempt in range(max_scaling_attempts):
            if attempt > 0:
                scaling_factor = scaling_multiplier ** attempt

            scaled_feed_for_lle = {}
            scaled_feed_for_lle[solvent] = base_solvent_amount * scaling_factor

            for compound, amount in non_liquid_compounds.items():
                scaled_feed_for_lle[compound] = amount

            scaled_excess = excess_compound * scaling_factor

            try:
                lle_excess, is_reliable, solubility_ratio = solve_lle_with_consensus(
                    scaled_feed_dict=scaled_feed_for_lle,
                    scaled_excess_amount=scaled_excess,
                    target_compound=target_compound,
                    temperature=temperature,
                    solvent=solvent,
                    num_attempts=3,
                    max_deviation=3.0
                )

                final_lle_excess = lle_excess
                final_solubility_ratio = solubility_ratio
                final_is_reliable = is_reliable

            except Exception as e:
                if verbose:
                    print(f"LLE consensus check failed: {e}")
                continue

            total_mol_l = sum(lle_excess.imol['l', chem] for chem in chemicals)
            total_mol_L = sum(lle_excess.imol['L', chem] for chem in chemicals)

            total_mol = total_mol_l + total_mol_L
            min_phase_fraction = 0.01

            if total_mol_l >= min_phase_fraction * total_mol and total_mol_L >= min_phase_fraction * total_mol:
                phase_separation_achieved = True
                if verbose and scaling_factor > 1:
                    print(f"Phase separation achieved at {scaling_factor:.0f}x scaling")
                break

        if not phase_separation_achieved:
            if verbose:
                print(f"Could not achieve phase separation for {target_compound}")
                print(f"Assuming fully soluble - all goes to EXTRACT")

            separation_status[target_compound] = 'Assumed_Fully_Soluble'
            total_residual_amounts[target_compound] = 0

            new_feed_state = current_feed_state.copy()
            new_feed_state[target_compound] = actual_compound_amount

            for compound, amount in non_liquid_compounds.items():
                new_feed_state[compound] = amount

            current_feed_state = new_feed_state
            continue

        lle_excess = final_lle_excess
        solubility_ratio = final_solubility_ratio
        is_reliable = final_is_reliable

        solvent_in_l = lle_excess.imol['l', solvent]
        solvent_in_L = lle_excess.imol['L', solvent]

        if solvent_in_l > solvent_in_L:
            extract_phase = 'l'
        else:
            extract_phase = 'L'

        dissolved_in_extract_scaled = lle_excess.imol[extract_phase, target_compound]

        if scaling_factor <= 1.0:
            dissolved_amount = min(dissolved_in_extract_scaled, actual_compound_amount)
            residual_amount = actual_compound_amount - dissolved_amount

            if is_reliable:
                separation_status[target_compound] = 'Calculated_Directly'
            else:
                separation_status[target_compound] = 'Calculated_Directly_Unreliable'

        else:
            max_dissolvable_at_current_solvent = solubility_ratio * base_solvent_amount

            if is_reliable:
                separation_status[target_compound] = 'Extrapolated'
            else:
                separation_status[target_compound] = 'Extrapolated_Unreliable'

            dissolved_amount = min(max_dissolvable_at_current_solvent, actual_compound_amount)
            residual_amount = actual_compound_amount - dissolved_amount

        total_residual_amounts[target_compound] = residual_amount

        new_feed_state = current_feed_state.copy()
        new_feed_state[target_compound] = dissolved_amount

        for compound, amount in non_liquid_compounds.items():
            new_feed_state[compound] = amount

        current_feed_state = new_feed_state

    extract_composition = current_feed_state
    residual_composition = total_residual_amounts

    return extract_composition, residual_composition, separation_status


# =============================================================================
#                    SOLID-SOLVENT EXTRACTOR UNIT
# =============================================================================

@cost('Solids loading',
      ID='Centrifuge',
      units='ton/hr',
      cost=68040,
      CE=567,
      n=0.50,
      BM=1.39,
      S=1)
class SolidSolventExtractor(Unit):
    """
    Continuous Solid-Solvent Extractor with Integrated Centrifuge and Kinetics

    A combined unit operation that performs solid-liquid equilibrium (SLE)
    and liquid-liquid equilibrium (LLE) separation in a stirred tank, followed
    by centrifugal separation of solids. Uses Reddy-Doraiswamy extraction
    kinetics to account for mass transfer limitations.

    For compounds without UNIFAC groups in thermosteam, the unit automatically
    falls back to Hansen Solubility Parameter (HSP) based predictions using
    the Flory-Huggins model from the Polymersolubility module.

    The unit is designed and costed as a combination of:

    1. **Stirred Mixing Tank** - for extraction contact and equilibrium
    2. **Solids Centrifuge** - for final solid-liquid separation

    **Equilibrium Models:**

    - **UNIFAC compounds:** Standard SLE/LLE via thermosteam activity coefficients
    - **Non-UNIFAC compounds:** Flory-Huggins with Hansen Solubility Parameters
        - Small molecules: N=1 (predict_compound_solubility)
        - Polymers: N=1000 default (predict_solubility)

    **Kinetics Model:**

    The Reddy-Doraiswamy correlation is used to calculate diffusion coefficients
    and extraction rates:

        Dab = K × (Ms^0.5 × T) / (μ × (VA×VB)^0.333)

    where K = 8.5e-8 if VA/VB < 1.5, else K = 1.0e-7

    The extraction follows:

        C(t) = t / (k1 + t/C_eq)

    where k1 = R² / (15 × π × Dab × X × Cs)

    Parameters
    ----------
    ins :
        [0] Feed stream (solid feedstock)
        [1] Solvent stream
    outs :
        [0] Solid/residual stream
        [1] Extract stream (solvent-rich with dissolved compounds)
    tau : float, optional
        Residence time in the mixing tank (hours). Default is 2 hours.
    V : float, optional
        Vessel volume in m³. If specified, overrides tau.
    solvent_ID : str, optional
        Chemical ID of the solvent. If not specified, will use dominant
        component in the solvent feed stream.
    reactor_type : str, optional
        Type of extraction reactor, which determines the accessibility factor.
        Valid options are:
        - 'conventional' (default): Standard stirred tank, accessibility = 0.67
        - 'ultrasound': Ultrasound-assisted extraction, accessibility = 0.86
        - 'microwave': Microwave-assisted extraction, accessibility = 1.0
        The accessibility factor multiplies the equilibrium concentration to
        account for how effectively the solvent can reach solute within the
        solid matrix.
    particle_radius : float, optional
        Particle radius in cm. Default is 0.254 cm.
    porosity_tortuosity_ratio : float, optional
        Ratio of porosity to tortuosity (ε/τ). Default is 0.13.
    sle_max_iter : int, optional
        Maximum iterations for SLE calculation. Default is 20.
    sle_tol : float, optional
        Convergence tolerance for SLE. Default is 0.001.
    lle_attempts : int, optional
        Number of LLE consensus attempts. Default is 3.
    verbose : bool, optional
        If True, print detailed separation progress. Default is False.

    Attributes
    ----------
    kinetics_results : dict
        Dictionary containing kinetics calculation results for each compound
    extraction_fractions : dict
        Dictionary mapping compound names to their extraction fraction achieved
    equilibrium_amounts : dict
        Equilibrium amounts before kinetics correction
    hsp_results : dict
        Dictionary containing HSP solubility results for non-UNIFAC compounds

    Notes
    -----
    **Kinetics Model Basis:**

    The Reddy-Doraiswamy correlation was validated against experimental
    beta-carotene extraction data from carrots at 30-50°C using ethanol
    as solvent. The model achieved R² > 0.99 across all temperatures.

    **HSP Fallback:**

    Compounds are automatically routed to HSP-based solubility prediction if
    they lack UNIFAC group definitions. The method distinguishes between:
    - Polymers (cellulose, lignin, etc.): Uses Flory-Huggins with N>>1
    - Small molecules: Uses Flory-Huggins with N=1

    **Cost Basis:**

    - **Mixing Tank (conventional):** Based on Apostolakou et al. (2009)
        - Base cost: $12,080 at 10 m³
        - Scaling exponent: 0.525
        - Bare module: 2.5

    - **Ultrasound Extractor (UAE):**
        - Base cost: $68,800 at 3.6 kW (CEPCI 798.88, 2025)
        - Scaling exponent: 1.0
        - Power: 1.625 kW per L/min of total flow

    - **Microwave Extractor (MAE):**
        - Base cost: $180,000 at 100 kW (CEPCI 596.2, 2020)
        - Scaling exponent: 0.6
        - Power: 12.5 kW per L/min of total flow

    - **Centrifuge:** Based on Seider et al. (2017)
        - Base cost: $68,040 at 1 ton/hr solids
        - Scaling exponent: 0.50
        - Bare module: 1.39

    **Power Requirements:**

    - Conventional tank agitation: 0.5 kW per m³ of vessel volume
    - UAE: 1.625 kW per L/min of total flow
    - MAE: 12.5 kW per L/min of total flow
    - Centrifuge: 1.4 kW per m³/hr of throughput

    Examples
    --------
    >>> import biosteam as bst
    >>> import thermosteam as tmo
    >>>
    >>> chemicals = tmo.Chemicals(['lycopene', 'glucose', 'ethanol', 'water'])
    >>> tmo.settings.set_thermo(chemicals)
    >>>
    >>> feed = bst.Stream('feed', lycopene=10, glucose=20, water=5,
    ...                   T=300, units='kg/hr')
    >>> solvent = bst.Stream('solvent', ethanol=100, T=300, units='kg/hr')
    >>>
    >>> E1 = SolidSolventExtractor('E1',
    ...                            ins=[feed, solvent],
    ...                            outs=['solid', 'extract'],
    ...                            tau=2,
    ...                            solvent_ID='ethanol',
    ...                            reactor_type='ultrasound',  # or 'conventional', 'microwave'
    ...                            particle_radius=0.25)
    >>>
    >>> E1.simulate()
    >>> print(E1.extraction_fractions)
    >>> print(f"Accessibility factor: {E1.accessibility_factor}")
    """

    _N_ins = 2
    _N_outs = 2
    _units = {'Volume': 'm^3',
              'Residence time': 'hr',
              'Solids loading': 'ton/hr',
              'Extractor power': 'kW',
              'Tank power': 'kW',
              'Centrifuge power': 'kW',
              'Total power': 'kW'}

    _has_power_utility = True

    # Accessibility factors for different reactor types
    # These factors account for how effectively the solvent can access
    # the solute within the solid matrix
    ACCESSIBILITY_FACTORS = {
        'conventional': 0.67,
        'ultrasound': 0.86,
        'microwave': 1.0
    }

    def __init__(self, ID='', ins=None, outs=(), thermo=None,
                 tau=2.0,
                 V=None,
                 solvent_ID=None,
                 reactor_type='conventional',
                 particle_radius=0.254,
                 porosity_tortuosity_ratio=0.13,
                 sle_max_iter=20,
                 sle_tol=0.001,
                 lle_attempts=3,
                 verbose=False):

        Unit.__init__(self, ID, ins, outs, thermo)

        self.tau = tau
        self.V = V
        self.solvent_ID = solvent_ID
        self.reactor_type = reactor_type
        self.particle_radius = particle_radius
        self.porosity_tortuosity_ratio = porosity_tortuosity_ratio
        self.sle_max_iter = sle_max_iter
        self.sle_tol = sle_tol
        self.lle_attempts = lle_attempts
        self.verbose = verbose

        # Results storage
        self.kinetics_results = {}
        self.extraction_fractions = {}
        self.equilibrium_amounts = {}
        self.hsp_results = {}  # Store HSP solubility results for non-UNIFAC compounds

    @property
    def reactor_type(self):
        """Reactor type for extraction (conventional, ultrasound, or microwave)."""
        return self._reactor_type

    @reactor_type.setter
    def reactor_type(self, value):
        """Set reactor type with validation."""
        value_lower = value.lower().strip()
        # Handle common variations in naming
        if value_lower in ('ultrasound', 'ultrasound assisted', 'ultrasound-assisted', 'us'):
            value_lower = 'ultrasound'
        elif value_lower in ('microwave', 'microwave assisted', 'microwave-assisted', 'mw'):
            value_lower = 'microwave'
        elif value_lower in ('conventional', 'standard', 'normal'):
            value_lower = 'conventional'

        if value_lower not in self.ACCESSIBILITY_FACTORS:
            valid_types = list(self.ACCESSIBILITY_FACTORS.keys())
            raise ValueError(
                f"Invalid reactor_type '{value}'. "
                f"Valid options are: {valid_types}"
            )
        self._reactor_type = value_lower

    @property
    def accessibility_factor(self):
        """
        Get the accessibility factor for the current reactor type.

        The accessibility factor accounts for how effectively the solvent
        can access the solute within the solid matrix:
        - Conventional: 0.67 (limited by diffusion through pores)
        - Ultrasound assisted: 0.86 (cavitation improves mass transfer)
        - Microwave assisted: 1.0 (volumetric heating maximizes accessibility)

        Returns
        -------
        float
            Accessibility factor (0-1)
        """
        return self.ACCESSIBILITY_FACTORS[self._reactor_type]

    def _get_solvent_properties(self, solvent_chem, T, P=101325):
        """
        Get solvent properties needed for kinetics calculations.

        Parameters
        ----------
        solvent_chem : thermosteam.Chemical
            Solvent chemical object
        T : float
            Operating temperature (K)
        P : float, optional
            Operating pressure (Pa). Default is 101325 Pa.
        """
        MW = solvent_chem.MW
        Tb = solvent_chem.Tb

        # Molar volume at boiling point - Tb is defined at atmospheric pressure
        # so 101325 Pa is correct here for the Reddy-Doraiswamy correlation
        try:
            Molvol_Tb = solvent_chem.V.l(Tb, 101325) * 1e6
        except:
            Molvol_Tb = MW / 0.8

        # Viscosity at operating temperature and pressure
        try:
            viscosity = solvent_chem.mu.l(T, P) * 1000
        except:
            viscosity = 0.5

        return {
            'MW': MW,
            'Tb': Tb,
            'Molvol_Tb': Molvol_Tb,
            'viscosity': viscosity
        }

    def _get_solute_molvol_Tb(self, solute_chem):
        """
        Get solute molar volume at its boiling point.

        Note: Uses atmospheric pressure (101325 Pa) because boiling point
        is defined at 1 atm. This is correct for the Reddy-Doraiswamy correlation.
        """
        Tb = solute_chem.Tb
        if Tb is None:
            return solute_chem.MW / 1.0

        try:
            return solute_chem.V.l(Tb, 101325) * 1e6
        except:
            return solute_chem.MW / 1.0

    def _apply_kinetics(self, compound_name, equilibrium_amount, total_feed_amount,
                        T, solvent_props):
        """
        Apply Reddy-Doraiswamy kinetics to determine actual extraction amount.

        The equilibrium amount is adjusted by the accessibility factor based on
        reactor type before being used in the kinetic model:
        - Conventional: 0.67 (limited by diffusion through pores)
        - Ultrasound assisted: 0.86 (cavitation improves mass transfer)
        - Microwave assisted: 1.0 (volumetric heating maximizes accessibility)
        """
        solute_chem = self.chemicals[compound_name]
        Molvol_solute_Tb = self._get_solute_molvol_Tb(solute_chem)

        Dab, K_used, VA_VB = calculate_Dab_RD(
            T=T,
            viscosity=solvent_props['viscosity'],
            Molvol_solute_Tb=Molvol_solute_Tb,
            Molvol_solvent_Tb=solvent_props['Molvol_Tb'],
            molmassS=solvent_props['MW']
        )

        Cs = total_feed_amount if total_feed_amount > 0 else 1.0

        k1 = calculate_k1_from_Dab(
            Dab=Dab,
            ParticleR=self.particle_radius,
            X=self.porosity_tortuosity_ratio,
            Cs=Cs
        )

        t_seconds = self.tau * 3600

        # Apply accessibility factor to equilibrium amount
        # This represents the maximum amount that can be extracted given
        # the reactor type's ability to access solute within the solid matrix
        accessible_equilibrium = equilibrium_amount * self.accessibility_factor

        if accessible_equilibrium > 0:
            extraction_frac = calculate_extraction_fraction(
                t=t_seconds,
                k1=k1,
                EQcomp=accessible_equilibrium
            )
        else:
            extraction_frac = 1.0

        actual_amount = accessible_equilibrium * extraction_frac

        kinetics_data = {
            'Dab': Dab,
            'K_used': K_used,
            'VA_VB': VA_VB,
            'k1': k1,
            'Molvol_solute_Tb': Molvol_solute_Tb,
            't_seconds': t_seconds,
            'equilibrium_amount': equilibrium_amount,
            'accessible_equilibrium': accessible_equilibrium,
            'accessibility_factor': self.accessibility_factor,
            'reactor_type': self.reactor_type,
            'extraction_fraction': extraction_frac,
            'actual_amount': actual_amount
        }

        return actual_amount, kinetics_data

    def _run(self):
        """
        Run mass and energy balance using solid-solvent extraction equilibrium
        with Reddy-Doraiswamy kinetics correction.
        """
        feed, solvent_stream = self.ins
        solid_out, extract_out = self.outs

        # Reset results storage
        self.kinetics_results = {}
        self.extraction_fractions = {}
        self.equilibrium_amounts = {}
        self.hsp_results = {}

        P = solvent_stream.P

        # Calculate mixing temperature
        feed_Cp = 0
        for chem_id in feed.chemicals.IDs:
            mol_flow = feed.imol[chem_id]
            if mol_flow > 1e-10:
                chem = self.chemicals[chem_id]
                try:
                    if hasattr(chem, 'Cn') and callable(chem.Cn):
                        Cp = chem.Cn(feed.T)
                    elif hasattr(chem, 'Cl') and callable(chem.Cl):
                        Cp = chem.Cl(feed.T)
                    else:
                        Cp = None

                    if Cp is not None:
                        feed_Cp += mol_flow * Cp
                except:
                    pass

        solvent_Cp = 0
        for chem_id in solvent_stream.chemicals.IDs:
            mol_flow = solvent_stream.imol[chem_id]
            if mol_flow > 1e-10:
                chem = self.chemicals[chem_id]
                try:
                    if hasattr(chem, 'Cl') and callable(chem.Cl):
                        Cp = chem.Cl(solvent_stream.T)
                        solvent_Cp += mol_flow * Cp
                except:
                    pass

        if feed_Cp > 0 and solvent_Cp > 0:
            T = (feed.T * feed_Cp + solvent_stream.T * solvent_Cp) / (feed_Cp + solvent_Cp)
        else:
            T = (feed.T * feed.F_mass + solvent_stream.T * solvent_stream.F_mass) / (
                        feed.F_mass + solvent_stream.F_mass)

        if self.solvent_ID is None:
            self.solvent_ID = solvent_stream.get_main_chemical()

        solvent = self.solvent_ID

        solvent_chem = self.chemicals[solvent]
        solvent_props = self._get_solvent_properties(solvent_chem, T, P)

        # Combine feed and solvent streams
        feed_dict = {}

        for chem in feed.chemicals.IDs:
            amount = feed.imol[chem]
            if amount > 1e-10:
                feed_dict[chem] = feed_dict.get(chem, 0) + amount

        for chem in solvent_stream.chemicals.IDs:
            amount = solvent_stream.imol[chem]
            if amount > 1e-10:
                feed_dict[chem] = feed_dict.get(chem, 0) + amount

        total_feed_amounts = feed_dict.copy()

        if solvent not in feed_dict or feed_dict[solvent] < 1e-10:
            raise ValueError(f"Solvent '{solvent}' not found in feed streams")

        # ======================================================================
        # STEP 0: IDENTIFY COMPOUNDS AND CHECK UNIFAC GROUPS
        # ======================================================================
        if self.verbose:
            print("\n" + "=" * 80)
            print("STEP 0: CLASSIFYING COMPOUNDS")
            print("=" * 80)

        # Classify compounds by phase AND UNIFAC availability
        unifac_liquid_compounds = {}
        unifac_non_liquid_compounds = {}
        non_unifac_compounds = {}

        for compound_name, amount in feed_dict.items():
            if compound_name == solvent:
                continue

            chemical = self.chemicals[compound_name]
            Tm = chemical.Tm if hasattr(chemical, 'Tm') else None
            Tb = chemical.Tb if hasattr(chemical, 'Tb') else None

            # Check if compound has UNIFAC groups
            has_unifac = has_unifac_groups(chemical)

            # Determine phase at operating temperature
            is_liquid = False
            if Tm and Tb:
                if Tm < T < Tb:
                    is_liquid = True

            if not has_unifac:
                # No UNIFAC groups - will use HSP solubility
                non_unifac_compounds[compound_name] = amount
                if self.verbose:
                    phase_str = "liquid" if is_liquid else "non-liquid"
                    print(f"  {compound_name}: No UNIFAC groups ({phase_str}) → HSP solubility")
            elif is_liquid:
                unifac_liquid_compounds[compound_name] = amount
                if self.verbose:
                    print(f"✓ {compound_name}: UNIFAC + LIQUID at {T:.1f}K → LLE")
            else:
                unifac_non_liquid_compounds[compound_name] = amount
                if self.verbose:
                    print(f"✓ {compound_name}: UNIFAC + non-liquid at {T:.1f}K → SLE")

        solvent_amount = feed_dict[solvent]

        # ======================================================================
        # STEP 0.5: PROCESS NON-UNIFAC COMPOUNDS VIA HSP SOLUBILITY
        # ======================================================================
        hsp_liquid_amounts = {}
        hsp_solid_amounts = {}
        hsp_results = {}

        if non_unifac_compounds:
            if self.verbose:
                print("\n" + "=" * 80)
                print("STEP 0.5: HSP SOLUBILITY FOR NON-UNIFAC COMPOUNDS")
                print("=" * 80)
                print(f"Processing {len(non_unifac_compounds)} compound(s) without UNIFAC groups...")

            for compound_name, amount in non_unifac_compounds.items():
                chemical = self.chemicals[compound_name]

                hsp_result = calculate_hsp_solubility_extraction(
                    compound_name=compound_name,
                    solvent_name=solvent,
                    compound_feed_mol=amount,
                    solvent_mol=solvent_amount,
                    temperature=T,
                    chemical=chemical,
                    pressure=P
                )

                hsp_results[compound_name] = hsp_result
                hsp_liquid_amounts[compound_name] = hsp_result['dissolved_mol']
                hsp_solid_amounts[compound_name] = hsp_result['solid_mol']

                if self.verbose:
                    if hsp_result['success']:
                        print(f"  {compound_name} ({hsp_result['method']}):")
                        print(f"    Solubility: {hsp_result['solubility_g_L']:.2f} g/L")
                        print(f"    Dissolved: {hsp_result['dissolved_mol']:.4f} mol")
                        print(f"    Solid: {hsp_result['solid_mol']:.4f} mol")
                    else:
                        print(f"  {compound_name}: {hsp_result['error']} → assumed fully soluble")

            # Store HSP results for later access
            self.hsp_results = hsp_results

        # ======================================================================
        # STEP 1: SOLID-LIQUID EQUILIBRIUM (SLE) - UNIFAC COMPOUNDS ONLY
        # ======================================================================
        if self.verbose:
            print("\n" + "=" * 80)
            print("STEP 1: SOLID-LIQUID EQUILIBRIUM (SLE)")
            print("=" * 80)

        if unifac_non_liquid_compounds:
            if self.verbose:
                print(f"Processing {len(unifac_non_liquid_compounds)} non-liquid UNIFAC compounds...")

            liquid_amounts, solid_amounts, num_iterations = calculate_multicomponent_sle_iterative(
                solute_feeds=unifac_non_liquid_compounds,
                solvent=solvent,
                solvent_flow=solvent_amount,
                temperature=T,
                max_iter=self.sle_max_iter,
                tol=self.sle_tol
            )

            if self.verbose:
                print(f"SLE converged in {num_iterations} iterations")
                print(f"Dissolved in liquid phase: {sum(liquid_amounts.values()):.4f} mol")
                print(f"Remaining as solid: {sum(solid_amounts.values()):.4f} mol")
        else:
            if self.verbose:
                print("No non-liquid UNIFAC compounds - skipping SLE")
            liquid_amounts = {}
            solid_amounts = {}

        # Merge HSP solid amounts into solid_amounts
        for compound, amount in hsp_solid_amounts.items():
            if amount > 1e-10:
                solid_amounts[compound] = solid_amounts.get(compound, 0) + amount

        # Build liquid feed for LLE (UNIFAC compounds only)
        liquid_feed_for_lle = {solvent: solvent_amount}
        liquid_feed_for_lle.update(liquid_amounts)
        liquid_feed_for_lle.update(unifac_liquid_compounds)

        # ======================================================================
        # STEP 2: LIQUID-LIQUID EQUILIBRIUM (LLE) - UNIFAC COMPOUNDS ONLY
        # ======================================================================
        if self.verbose:
            print("\n" + "=" * 80)
            print("STEP 2: LIQUID-LIQUID EQUILIBRIUM (LLE)")
            print("=" * 80)
            print("Performing sequential LLE on UNIFAC compounds in liquid phase...")

        # Only run LLE if there are non-solvent compounds
        non_solvent_lle_compounds = {k: v for k, v in liquid_feed_for_lle.items() if k != solvent}

        if non_solvent_lle_compounds:
            extract_composition, residual_composition, separation_status = sequential_lle_separation(
                feed=liquid_feed_for_lle,
                temperature=T,
                solvent=solvent,
                thermo=self.thermo,
                verbose=self.verbose
            )
        else:
            if self.verbose:
                print("No UNIFAC compounds for LLE - skipping")
            extract_composition = {solvent: solvent_amount}
            residual_composition = {}
            separation_status = {}

        # Merge HSP-calculated liquid amounts into extract composition
        for compound, amount in hsp_liquid_amounts.items():
            if amount > 1e-10:
                extract_composition[compound] = extract_composition.get(compound, 0) + amount

        self.equilibrium_amounts = extract_composition.copy()

        # ======================================================================
        # STEP 3: APPLY KINETICS CORRECTION TO ALL COMPOUNDS
        # ======================================================================
        if self.verbose:
            print("\n" + "=" * 80)
            print("STEP 3: APPLYING REDDY-DORAISWAMY KINETICS")
            print("=" * 80)
            print(f"Reactor type: {self.reactor_type}")
            print(f"Accessibility factor: {self.accessibility_factor}")
            print(f"Particle radius: {self.particle_radius} cm")
            print(f"Porosity/tortuosity ratio: {self.porosity_tortuosity_ratio}")
            print(f"Residence time: {self.tau} hours ({self.tau * 3600:.0f} seconds)")
            print(f"Solvent viscosity at {T:.1f}K: {solvent_props['viscosity']:.4f} cP")
            print("-" * 80)

        kinetics_corrected_extract = {}

        for compound, eq_amount in extract_composition.items():
            if compound == solvent:
                kinetics_corrected_extract[compound] = eq_amount
                continue

            total_feed = total_feed_amounts.get(compound, eq_amount)

            actual_amount, kinetics_data = self._apply_kinetics(
                compound_name=compound,
                equilibrium_amount=eq_amount,
                total_feed_amount=total_feed,
                T=T,
                solvent_props=solvent_props
            )

            kinetics_corrected_extract[compound] = actual_amount
            self.kinetics_results[compound] = kinetics_data
            self.extraction_fractions[compound] = kinetics_data['extraction_fraction']

            unextracted = eq_amount - actual_amount
            if unextracted > 1e-10:
                if compound in solid_amounts:
                    solid_amounts[compound] += unextracted
                else:
                    solid_amounts[compound] = unextracted

                if compound in residual_composition:
                    residual_composition[compound] += unextracted
                else:
                    residual_composition[compound] = unextracted

            if self.verbose:
                frac = kinetics_data['extraction_fraction']
                acc_eq = kinetics_data['accessible_equilibrium']
                print(f"{compound}:")
                print(f"  Dab = {kinetics_data['Dab']:.4e} cm²/s")
                print(f"  k1 = {kinetics_data['k1']:.2f}")
                print(f"  Equilibrium: {eq_amount:.4f} mol")
                print(f"  Accessible equilibrium: {acc_eq:.4f} mol ({self.accessibility_factor*100:.0f}%)")
                print(f"  Extracted: {actual_amount:.4f} mol ({frac*100:.1f}% of accessible equilibrium)")

        extract_composition = kinetics_corrected_extract

        if self.verbose:
            print("\n" + "=" * 80)
            print("SEPARATION COMPLETE")
            print("=" * 80)
            print(f"\nExtraction fractions achieved:")
            for comp, frac in self.extraction_fractions.items():
                print(f"  {comp}: {frac*100:.1f}%")

            unreliable_compounds = [k for k, v in separation_status.items() if 'Unreliable' in v]
            if unreliable_compounds:
                print(f"\n⚠️  Warning: {len(unreliable_compounds)} compound(s) with unreliable separation:")
                for comp in unreliable_compounds:
                    print(f"  - {comp}: {separation_status[comp]}")

        # ======================================================================
        # SET OUTPUT STREAMS
        # ======================================================================
        solid_out.phase = 's'
        solid_out.T = T
        solid_out.P = P
        solid_out.imol.data[:] = 0

        for compound, amount in solid_amounts.items():
            if amount > 1e-10:
                solid_out.imol[compound] = amount

        for compound, amount in residual_composition.items():
            if amount > 1e-10 and compound not in solid_amounts:
                solid_out.imol[compound] += amount

        extract_out.phase = 'l'
        extract_out.T = T
        extract_out.P = P
        extract_out.imol.data[:] = 0

        for compound, amount in extract_composition.items():
            if amount > 1e-10:
                extract_out.imol[compound] = amount

    # Cost correlation parameters for each reactor type
    # For conventional: mixing tank scaled on Volume (m³)
    # For UAE/MAE: extractor equipment scaled on power (kW)
    EXTRACTOR_COST_PARAMS = {
        'conventional': {
            'basis': 'Volume',
            'units': 'm^3',
            'cost': 12080,
            'CE': 525.4,
            'n': 0.525,
            'BM': 2.5,
            'S': 10,
        },
        'ultrasound': {
            'basis': 'Extractor power',
            'units': 'kW',
            'cost': 68800,
            'CE': 798.88,
            'n': 1.0,
            'BM': 2.5,
            'S': 3.6,
        },
        'microwave': {
            'basis': 'Extractor power',
            'units': 'kW',
            'cost': 180000,
            'CE': 596.2,
            'n': 0.6,
            'BM': 2.5,
            'S': 100,
        },
    }

    def _design(self):
        """Design the extractor vessel and centrifuge."""
        Design = self.design_results

        feed, solvent = self.ins
        solid_out, extract_out = self.outs

        Q_feed = feed.F_vol
        Q_solvent = solvent.F_vol
        Q_total = Q_feed + Q_solvent

        if self.V is None:
            V = Q_total * self.tau
        else:
            V = self.V
            self.tau = V / Q_total if Q_total > 0 else 0

        V = max(V, 0.1)

        Design['Volume'] = V
        Design['Residence time'] = self.tau

        solids_mass_flow = solid_out.F_mass
        solids_loading = solids_mass_flow / 1000
        solids_loading = max(solids_loading, 0.1)

        Design['Solids loading'] = solids_loading

        # Power calculations depend on reactor type
        centrifuge_power = Q_total * 1.40

        if self._reactor_type == 'conventional':
            # Standard stirred tank: 0.5 kW per m³
            tank_power = V * 0.5
            Design['Extractor power'] = 0  # Not used for costing
        elif self._reactor_type == 'ultrasound':
            # UAE power: 1.625 kW per L/min of total flow
            Q_total_l_min = Q_total * 1000 / 60  # m³/hr -> L/min
            tank_power = 1.625 * Q_total_l_min
            Design['Extractor power'] = tank_power
        elif self._reactor_type == 'microwave':
            # MAE power: 12.5 kW per L/min of total flow
            Q_total_l_min = Q_total * 1000 / 60  # m³/hr -> L/min
            tank_power = 12.5 * Q_total_l_min
            Design['Extractor power'] = tank_power

        total_power = tank_power + centrifuge_power
        Design['Tank power'] = tank_power
        Design['Centrifuge power'] = centrifuge_power
        Design['Total power'] = total_power

        self.power_utility(total_power)

    def _cost(self):
        """Calculate equipment costs based on reactor type.

        The centrifuge cost is handled by the @cost decorator.
        The extractor/tank cost is calculated here based on reactor type.
        """
        # Determine equipment name and register bare-module factor
        # before any cost assignments to avoid BioSTEAM warnings
        if self._reactor_type == 'conventional':
            equip_name = 'Mixing tank'
        elif self._reactor_type == 'ultrasound':
            equip_name = 'Ultrasound extractor'
        elif self._reactor_type == 'microwave':
            equip_name = 'Microwave extractor'

        params = self.EXTRACTOR_COST_PARAMS[self._reactor_type]
        self.F_BM[equip_name] = 2.5

        # Let the @cost decorator handle the centrifuge
        super()._cost()

        # Now add the extractor/tank cost
        Design = self.design_results

        # Get the design value for costing basis
        design_value = Design[params['basis']]
        design_value = max(design_value, 1e-6)  # Avoid zero division

        # Scale the reference cost: C = C_ref * (S / S_ref)^n
        S_ref = params['S']
        C_ref = params['cost']
        n = params['n']

        scaled_cost = C_ref * (design_value / S_ref) ** n

        # Adjust for CEPCI
        current_CE = bst.CE
        ref_CE = params['CE']
        adjusted_cost = scaled_cost * (current_CE / ref_CE)

        self.baseline_purchase_costs[equip_name] = scaled_cost
        self.purchase_costs[equip_name] = adjusted_cost


# =============================================================================
# EXAMPLE USAGE AND TESTING
# =============================================================================
if __name__ == "__main__":
    import biosteam as bst
    import thermosteam as tmo

    print("=" * 80)
    print("SOLID-SOLVENT EXTRACTOR WITH KINETICS - EXAMPLE USAGE")
    print("=" * 80)

    # Define chemicals
    chemicals = tmo.Chemicals(['lycopene', 'tyrosol', 'ethanol', 'hexane',
                               'glucose', 'palmitic_acid', 'citric_acid',
                               'water', 'chloroform', 'beta-carotene'])

    thermo = tmo.Thermo(
        chemicals=chemicals,
        mixture=tmo.RKMixture.from_chemicals(chemicals),
        Gamma=tmo.DortmundActivityCoefficients,
    )
    tmo.settings.set_thermo(thermo)
    bst.settings.set_thermo(thermo)

    bst.CE = 607.5

    print("\n" + "-" * 80)
    print("CREATING STREAMS")
    print("-" * 80)

    feed = bst.Stream('feed',
                      lycopene=10,
                      glucose=20,
                      citric_acid=15,
                      palmitic_acid=5,
                      beta_carotene=8,
                      water=10,
                      T=323.15,
                      units='kg/hr')

    print(f"\nFeed stream:")
    feed.show()

    solvent = bst.Stream('solvent',
                         ethanol=100,
                         T=323.15,
                         units='kg/hr')

    print(f"\nSolvent stream:")
    solvent.show()

    # ==========================================================================
    # RUN EXTRACTION
    # ==========================================================================
    print("\n" + "=" * 80)
    print("RUNNING EXTRACTION WITH KINETICS")
    print("=" * 80)

    E1 = SolidSolventExtractor('E1',
                                ins=[feed, solvent],
                                outs=['solid', 'extract'],
                                tau=2.0,
                                solvent_ID='ethanol',
                                reactor_type='conventional',
                                particle_radius=0.254,
                                porosity_tortuosity_ratio=0.13,
                                verbose=True)

    E1.simulate()

    print("\n✓ Simulation complete!")

    print("\n" + "-" * 80)
    print("OUTPUT STREAMS")
    print("-" * 80)

    print("\nSolid/Residual stream:")
    E1.outs[0].show()

    print("\nExtract stream:")
    E1.outs[1].show()

    # ==========================================================================
    # EFFECT OF RESIDENCE TIME
    # ==========================================================================
    print("\n" + "=" * 80)
    print("EFFECT OF RESIDENCE TIME ON EXTRACTION")
    print("=" * 80)

    residence_times = [0.5, 1.0, 2.0, 4.0, 8.0]

    print(f"\n{'Residence Time (hr)':<20}", end='')
    for chem in ['lycopene', 'glucose', 'beta_carotene']:
        print(f"{chem + ' (%)':<15}", end='')
    print()
    print("-" * 65)

    for tau in residence_times:
        E_test = SolidSolventExtractor(f'E_tau_{tau}',
                                        ins=[feed.copy(), solvent.copy()],
                                        outs=[f'solid_{tau}', f'extract_{tau}'],
                                        tau=tau,
                                        solvent_ID='ethanol',
                                        particle_radius=0.254,
                                        porosity_tortuosity_ratio=0.13,
                                        verbose=False)
        E_test.simulate()

        print(f"{tau:<20.1f}", end='')
        for chem in ['lycopene', 'glucose', 'beta_carotene']:
            frac = E_test.extraction_fractions.get(chem, 0) * 100
            print(f"{frac:<15.1f}", end='')
        print()

    print("\n" + "=" * 80)
    print("EFFECT OF REACTOR TYPE ON EXTRACTION")
    print("=" * 80)

    reactor_types = ['conventional', 'ultrasound', 'microwave']

    print(f"\n{'Reactor Type':<20}{'Factor':<10}", end='')
    for chem in ['lycopene', 'glucose', 'beta_carotene']:
        print(f"{chem + ' (%)':<15}", end='')
    print()
    print("-" * 75)

    for reactor in reactor_types:
        E_reactor = SolidSolventExtractor(f'E_{reactor}',
                                           ins=[feed.copy(), solvent.copy()],
                                           outs=[f'solid_{reactor}', f'extract_{reactor}'],
                                           tau=2.0,
                                           solvent_ID='ethanol',
                                           reactor_type=reactor,
                                           particle_radius=0.254,
                                           porosity_tortuosity_ratio=0.13,
                                           verbose=False)
        E_reactor.simulate()

        print(f"{reactor:<20}{E_reactor.accessibility_factor:<10.2f}", end='')
        for chem in ['lycopene', 'glucose', 'beta_carotene']:
            frac = E_reactor.extraction_fractions.get(chem, 0) * 100
            print(f"{frac:<15.1f}", end='')
        print()

    print("\n" + "=" * 80)
    print("EXAMPLE COMPLETE")
    print("=" * 80)