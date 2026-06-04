"""
BioSTEAM Proxy Compound Finder
================================
For compounds not in the thermosteam/chemicals database, this module:

1. Checks if a compound can be loaded directly by thermosteam
   (tries name, then CAS number)
2. If not, resolves a SMILES string via multiple sources:
   name → CAS → ChEBI → PubChem REST → NIH CIR
3. Estimates HSP, density, MW from SMILES via Polymersolubility.py
4. Gets boiling point from the chemicals library or Joback estimation
5. Screens the thermosteam-compatible pool for the best proxy match
6. Returns a ready-to-use thermosteam Chemical with the proxy's thermo data

Compounds can be specified as plain strings or as dicts with optional
CAS and ChEBI identifiers:

    compounds = [
        'Water',                                          # simple name
        {'name': 'Syringaldehyde', 'cas': '134-96-3'},    # name + CAS
        {'name': 'Cyanidin-3-glucoside',                  # name + ChEBI + SMILES
         'chebi': 'CHEBI:80159',
         'smiles': 'OC1C(OC2=CC3=...'},
    ]

All HSP, density, and MW estimation is handled by Polymersolubility.py.
This module adds only the thermosteam screening and matching logic.

Requirements:
    pip install thermosteam rdkit
    Polymersolubility.py must be importable (same directory or on sys.path)
"""

import warnings
import math
import json
import os
import hashlib
import urllib.request
import urllib.parse
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Union

import thermosteam as tmo
from chemicals.identifiers import pubchem_db

# ─────────────────────────────────────────────────────────────────────
# Import from YOUR Polymersolubility module.
#
# Property estimation (Hoftyzer-Van Krevelen group contribution):
#   - estimate_hsp_from_smiles(smiles) → HSP(dD, dP, dH)
#   - estimate_density_from_smiles(smiles) → (density, source_str)
#   - HSP dataclass
#
# SMILES lookup chain (tries multiple sources in order):
#   - lookup_smiles(name) → (smiles, source)
#     Search order:
#       1. BiopolymerSMILESLookup (PubChem + ChEBI via OLS API)
#       2. PubChem REST API
#       3. NIH Chemical Identifier Resolver (CIR)
#   - lookup_smiles_from_pubchem(name) → smiles or None
#   - lookup_smiles_from_cir(name) → smiles or None
#   - is_valid_smiles(text) → bool
#
# Nothing is duplicated here — all logic lives in Polymersolubility.py.
# ─────────────────────────────────────────────────────────────────────
from Polymersolubility import (
    estimate_hsp_from_smiles,
    estimate_density_from_smiles,
    HSP,
    RDKIT_AVAILABLE,
    lookup_smiles,
    lookup_smiles_from_pubchem,
    lookup_smiles_from_cir,
    is_valid_smiles,
    # Group contribution data + SMARTS matching (Van Krevelen 2009)
    # Used here for Tm, Hfus, Vm estimation
    GROUP_CONTRIBUTIONS,
    _match_smarts_groups,
)

if RDKIT_AVAILABLE:
    from rdkit import Chem
    from rdkit.Chem import Descriptors


# ═══════════════════════════════════════════════════════════════════════
# COMPOUND INPUT SPECIFICATION
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class CompoundInput:
    """
    Normalised compound specification.

    Accepts plain strings or dicts with optional identifiers::

        'Water'                                           # just a name
        {'name': 'Syringaldehyde', 'cas': '134-96-3'}    # name + CAS
        {'name': 'Cyanidin', 'chebi': 'CHEBI:80159'}     # name + ChEBI
        {'name': 'X', 'cas': '...', 'chebi': '...', 'smiles': '...'}
    """
    name: str
    cas: Optional[str] = None
    chebi: Optional[str] = None
    smiles: Optional[str] = None


def _normalise_compound(entry) -> CompoundInput:
    """Convert a string or dict into a CompoundInput."""
    if isinstance(entry, str):
        return CompoundInput(name=entry)
    if isinstance(entry, dict):
        return CompoundInput(
            name=entry.get('name', ''),
            cas=entry.get('cas'),
            chebi=entry.get('chebi'),
            smiles=entry.get('smiles'),
        )
    if isinstance(entry, CompoundInput):
        return entry
    raise TypeError(f"Expected str, dict, or CompoundInput, got {type(entry)}")


# ═══════════════════════════════════════════════════════════════════════
# SMILES LOOKUP VIA CAS / ChEBI  (PubChem REST API)
# ═══════════════════════════════════════════════════════════════════════

def _lookup_smiles_by_cas(cas: str) -> Optional[str]:
    """
    Look up SMILES from PubChem using a CAS registry number.

    PubChem endpoint: /compound/name/{CAS}/property/CanonicalSMILES/JSON
    (PubChem resolves CAS numbers as synonyms.)
    """
    if not cas:
        return None
    try:
        encoded = urllib.parse.quote(cas.strip())
        url = (f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/"
               f"name/{encoded}/property/CanonicalSMILES/JSON")
        req = urllib.request.Request(url, headers={'User-Agent': 'Python/ProxyFinder'})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            props = data.get('PropertyTable', {}).get('Properties', [])
            if props:
                return props[0].get('CanonicalSMILES')
    except Exception:
        pass
    return None


def _lookup_smiles_by_chebi(chebi: str) -> Optional[str]:
    """
    Look up SMILES from PubChem using a ChEBI identifier.

    Accepts 'CHEBI:12345' or just '12345'.  Tries PubChem name search
    with the full 'CHEBI:...' string.
    """
    if not chebi:
        return None
    chebi = chebi.strip()
    # Normalise to 'CHEBI:12345' format
    if not chebi.upper().startswith('CHEBI:'):
        chebi = f"CHEBI:{chebi}"
    try:
        encoded = urllib.parse.quote(chebi)
        url = (f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/"
               f"name/{encoded}/property/CanonicalSMILES/JSON")
        req = urllib.request.Request(url, headers={'User-Agent': 'Python/ProxyFinder'})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            props = data.get('PropertyTable', {}).get('Properties', [])
            if props:
                return props[0].get('CanonicalSMILES')
    except Exception:
        pass
    return None


def _resolve_chebi_to_identifiers(chebi: str) -> Dict[str, Optional[str]]:
    """
    Resolve a ChEBI ID to a compound name, CAS, and SMILES via PubChem.

    Returns dict with keys: 'name', 'cas', 'smiles' (any may be None).
    Useful for feeding resolved names/CAS into thermosteam.
    """
    result = {'name': None, 'cas': None, 'smiles': None}
    if not chebi:
        return result
    chebi = chebi.strip()
    if not chebi.upper().startswith('CHEBI:'):
        chebi = f"CHEBI:{chebi}"

    # Step 1: Get PubChem CID and SMILES from ChEBI
    cid = None
    try:
        encoded = urllib.parse.quote(chebi)
        url = (f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/"
               f"name/{encoded}/property/CID,CanonicalSMILES,IUPACName/JSON")
        req = urllib.request.Request(url, headers={'User-Agent': 'Python/ProxyFinder'})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            props = data.get('PropertyTable', {}).get('Properties', [])
            if props:
                cid = props[0].get('CID')
                result['smiles'] = props[0].get('CanonicalSMILES')
                result['name'] = props[0].get('IUPACName')
    except Exception:
        pass

    # Step 2: Get synonyms (first synonym is usually the common name,
    #         CAS numbers appear as synonyms matching \d+-\d+-\d+ pattern)
    if cid:
        try:
            url = (f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/"
                   f"cid/{cid}/synonyms/JSON")
            req = urllib.request.Request(url, headers={'User-Agent': 'Python/ProxyFinder'})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
                syns = (data.get('InformationList', {})
                            .get('Information', [{}])[0]
                            .get('Synonym', []))
                if syns:
                    # First synonym is usually the common name
                    result['name'] = syns[0]
                    # Find CAS (pattern: digits-digits-digits)
                    import re
                    for s in syns:
                        if re.fullmatch(r'\d{2,7}-\d{2}-\d', s):
                            result['cas'] = s
                            break
        except Exception:
            pass

    return result


# ═══════════════════════════════════════════════════════════════════════
# MW FROM SMILES  (just an RDKit one-liner, no custom logic)
# ═══════════════════════════════════════════════════════════════════════

def get_mw_from_smiles(smiles: str) -> Optional[float]:
    """Molecular weight straight from RDKit."""
    if not RDKIT_AVAILABLE:
        return None
    mol = Chem.MolFromSmiles(smiles)
    return Descriptors.MolWt(mol) if mol else None


# ═══════════════════════════════════════════════════════════════════════
# Tm, Hfus, Vm ESTIMATION  (Van Krevelen 2009, Tables 4.10 / 5.7 / 6.8)
# ═══════════════════════════════════════════════════════════════════════
#
# All three use the same GROUP_CONTRIBUTIONS table and _match_smarts_groups()
# function from Polymersolubility.py — no logic is duplicated.
#
#   Tm   = ΣYm_i / MW            [K]         (Table 6.8)
#   Hfus = ΣHm_i                 [kJ/mol]    (Table 5.7)
#   Vm   = ΣV_i                  [cm³/mol]   (Table 4.10)
#
# ═══════════════════════════════════════════════════════════════════════

def estimate_tm_from_smiles(smiles: str) -> Optional[float]:
    """
    Van Krevelen (2009) group-contribution melting point estimate.

    Tm = ΣYm_i / MW   [K]

    Reference: Properties of Polymers, 4th Ed., Chapter 6, Table 6.8
    """
    groups = _match_smarts_groups(smiles)
    if not groups:
        return None
    mw = get_mw_from_smiles(smiles)
    if not mw or mw <= 0:
        return None

    sum_Ym = 0.0
    for name, count in groups.items():
        if name in GROUP_CONTRIBUTIONS:
            sum_Ym += GROUP_CONTRIBUTIONS[name]['Ym'] * count

    if sum_Ym <= 0:
        return None
    return round(sum_Ym / mw, 1)


def estimate_hfus_from_smiles(smiles: str) -> Optional[float]:
    """
    Van Krevelen (2009) group-contribution heat of fusion estimate.

    Hfus = ΣHm_i   [kJ/mol]

    Reference: Properties of Polymers, 4th Ed., Chapter 5, Table 5.7
    """
    groups = _match_smarts_groups(smiles)
    if not groups:
        return None

    sum_Hm = 0.0
    for name, count in groups.items():
        if name in GROUP_CONTRIBUTIONS:
            sum_Hm += GROUP_CONTRIBUTIONS[name]['Hm'] * count

    return round(sum_Hm, 2) if sum_Hm != 0 else None


def estimate_vm_from_smiles(smiles: str) -> Optional[float]:
    """
    Van Krevelen (2009) group-contribution molar volume estimate.

    Vm = ΣV_i   [cm³/mol]

    Reference: Properties of Polymers, 4th Ed., Chapter 4, Table 4.10
    """
    groups = _match_smarts_groups(smiles)
    if not groups:
        return None

    sum_V = 0.0
    for name, count in groups.items():
        if name in GROUP_CONTRIBUTIONS:
            sum_V += GROUP_CONTRIBUTIONS[name]['V'] * count

    # Fallback if volume too small (poor group matching)
    if sum_V < 20:
        mw = get_mw_from_smiles(smiles)
        if mw:
            sum_V = mw * 0.95

    return round(sum_V, 1) if sum_V > 0 else None


def _is_organic(smiles: str) -> bool:
    """True if the molecule contains at least one carbon atom."""
    if not RDKIT_AVAILABLE:
        return True
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    return any(atom.GetAtomicNum() == 6 for atom in mol.GetAtoms())


# ═══════════════════════════════════════════════════════════════════════
# PROXY CANDIDATE POOL  (thermosteam-compatible chemicals)
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ProxyCandidate:
    """A thermosteam-loadable chemical with its matching properties."""
    name: str
    cas: str
    smiles: str
    MW: float
    Tb: float       # K  (from thermosteam, kept for reference)
    Tm: Optional[float] = None    # K  (Van Krevelen estimate)
    Hfus: Optional[float] = None  # kJ/mol  (Van Krevelen estimate)
    Vm: Optional[float] = None    # cm³/mol (Van Krevelen estimate)
    hsp: Optional[HSP] = None


POOL_CACHE_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '_proxy_pool_cache.json'
)

ASSIGN_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '_assign_cache'
)


def build_proxy_pool(
    max_candidates: int = 5000,
    verbose: bool = True,
) -> List[ProxyCandidate]:
    """
    Screen the chemicals/pubchem database for thermosteam-compatible
    organic compounds and estimate their HSP via Polymersolubility.

    Each candidate must:
      - have a SMILES string
      - be organic (contains carbon)
      - load in thermosteam with a valid boiling point

    HSP is estimated using estimate_hsp_from_smiles() from
    Polymersolubility.py (Hoftyzer-Van Krevelen method).

    Results are cached to '_proxy_pool_cache.json' so subsequent
    calls load instantly.
    """
    # ── Try loading from cache ──────────────────────────────────────
    if os.path.exists(POOL_CACHE_FILE):
        try:
            with open(POOL_CACHE_FILE) as f:
                data = json.load(f)

            # ① Reject old list-format caches (pre-v2) immediately
            if isinstance(data, list):
                if verbose:
                    print("[proxy-pool] Cache outdated (pre-v2 list format), rebuilding …")
                raise ValueError("old cache format — needs v2 dict wrapper")

            # ② Check version tag
            if data.get('version') != 3:
                if verbose:
                    print("[proxy-pool] Cache version mismatch (need v3 with gas-phase validation), rebuilding …")
                raise ValueError("cache version mismatch")

            # ③ Load the pool list
            pool_data = data['pool']
            pool = []
            for d in pool_data:
                hsp_data = d.pop('hsp')
                hsp = HSP(**hsp_data) if hsp_data else None
                pool.append(ProxyCandidate(**d, hsp=hsp))
            if verbose:
                print(f"[proxy-pool] Loaded {len(pool)} candidates from cache")
            return pool
        except Exception:
            pass  # rebuild

    # ── Build from scratch ──────────────────────────────────────────
    if verbose:
        print("[proxy-pool] Building proxy pool from chemicals database …")
        print("             (this takes ~30s on first run, cached after)")

    pool: List[ProxyCandidate] = []
    entries = list(pubchem_db.CAS_index.items())

    for i, (cas_int, entry) in enumerate(entries[:max_candidates]):
        name = entry.common_name
        smiles = entry.smiles
        if not smiles or not name:
            continue

        # Only organic compounds make sensible proxies
        if not _is_organic(smiles):
            continue

        # Must load in thermosteam with a finite boiling point and valid Hf
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                chem = tmo.Chemical(name)
                tb = chem.Tb
                mw = chem.MW
                hf = chem.Hf
                if tb is None or math.isnan(tb) or mw is None:
                    continue
                # Proxy must have valid Hf — otherwise 0*NaN poisons Hnet
                if hf is None or (isinstance(hf, float) and math.isnan(hf)):
                    continue
        except Exception:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    chem = tmo.Chemical(entry.CASs)
                    tb = chem.Tb
                    mw = chem.MW
                    hf = chem.Hf
                    if tb is None or math.isnan(tb) or mw is None:
                        continue
                    if hf is None or (isinstance(hf, float) and math.isnan(hf)):
                        continue
            except Exception:
                continue

        # Proxy candidates must have working gas-phase enthalpy so they
        # don't crash Gas_Enthalpy_Ref_Solid in the evaporator
        if not getattr(chem, 'locked_state', None) and tb is not None:
            _pool_gas_ok = True
            for _T in [tb + 10, tb + 50]:
                try:
                    _hval = chem.H('g', _T, 101325)
                    if _hval is None:
                        _pool_gas_ok = False
                        break
                except Exception:
                    _pool_gas_ok = False
                    break
            if not _pool_gas_ok:
                continue

        # Estimate properties using Polymersolubility's functions
        hsp = estimate_hsp_from_smiles(smiles)
        tm = estimate_tm_from_smiles(smiles)
        hfus = estimate_hfus_from_smiles(smiles)
        vm = estimate_vm_from_smiles(smiles)

        pool.append(ProxyCandidate(
            name=name, cas=entry.CASs, smiles=smiles,
            MW=round(mw, 2), Tb=round(tb, 2),
            Tm=tm, Hfus=hfus, Vm=vm, hsp=hsp,
        ))

        if verbose and (i + 1) % 500 == 0:
            print(f"  … screened {i+1}/{min(max_candidates, len(entries))}"
                  f"  ({len(pool)} valid so far)")

    if verbose:
        hsp_count = sum(1 for c in pool if c.hsp is not None)
        print(f"[proxy-pool] Done: {len(pool)} candidates"
              f" ({hsp_count} with HSP)")

    # ── Cache to disk ───────────────────────────────────────────────
    try:
        pool_data = []
        for c in pool:
            d = {
                'name': c.name, 'cas': c.cas, 'smiles': c.smiles,
                'MW': c.MW, 'Tb': c.Tb,
                'Tm': c.Tm, 'Hfus': c.Hfus, 'Vm': c.Vm,
            }
            if c.hsp:
                d['hsp'] = {'dD': c.hsp.dD, 'dP': c.hsp.dP,
                            'dH': c.hsp.dH, 'source': c.hsp.source}
            else:
                d['hsp'] = None
            pool_data.append(d)
        cache = {'version': 3, 'pool': pool_data}
        with open(POOL_CACHE_FILE, 'w') as f:
            json.dump(cache, f)
        if verbose:
            print(f"[proxy-pool] Cache saved (v3, gas-phase validated) → {POOL_CACHE_FILE}")
    except Exception:
        pass

    return pool


# ═══════════════════════════════════════════════════════════════════════
# PROXY MATCHING
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ProxyMatch:
    """Result of a proxy search for one target compound."""
    target_name: str
    target_smiles: Optional[str]
    target_MW: Optional[float]
    target_Tm: Optional[float]
    target_Hfus: Optional[float]
    target_Vm: Optional[float]
    target_hsp: Optional[HSP]
    in_thermosteam: bool
    thermosteam_id: Optional[str] = None   # the ID that actually loaded in thermosteam (for native compounds)
    smiles_source: Optional[str] = None
    proxy_name: Optional[str] = None
    proxy_cas: Optional[str] = None
    proxy_MW: Optional[float] = None
    proxy_Tm: Optional[float] = None
    proxy_Hfus: Optional[float] = None
    proxy_Vm: Optional[float] = None
    proxy_hsp: Optional[HSP] = None
    distance: Optional[float] = None
    top_candidates: List[dict] = field(default_factory=list)

    def summary(self) -> str:
        if self.in_thermosteam:
            return (f"  ✓  {self.target_name:30s}"
                    f" — available natively in thermosteam")
        if self.proxy_name is None:
            return (f"  ✗  {self.target_name:30s}"
                    f" — NO PROXY FOUND (SMILES unavailable or no match)")
        lines = [
            f"  →  {self.target_name:30s}"
            f" — proxy: {self.proxy_name} (CAS {self.proxy_cas})",
            f"     {'':30s}   distance = {self.distance:.4f}"
            f"   SMILES via: {self.smiles_source or 'unknown'}",
        ]
        if self.target_hsp:
            lines.append(
                f"     {'':30s}   target  MW={self.target_MW:>7.1f}"
                f"  Tm={self.target_Tm or 0:>6.1f} K"
                f"  Vm={self.target_Vm or 0:>6.1f}"
                f"  HSP({self.target_hsp.dD:.1f},"
                f" {self.target_hsp.dP:.1f},"
                f" {self.target_hsp.dH:.1f})"
            )
        if self.proxy_hsp:
            lines.append(
                f"     {'':30s}   proxy   MW={self.proxy_MW:>7.1f}"
                f"  Tm={self.proxy_Tm or 0:>6.1f} K"
                f"  Vm={self.proxy_Vm or 0:>6.1f}"
                f"  HSP({self.proxy_hsp.dD:.1f},"
                f" {self.proxy_hsp.dP:.1f},"
                f" {self.proxy_hsp.dH:.1f})"
            )
        return '\n'.join(lines)


def _normalised_distance(
    target_MW: float, target_Tm: float,
    target_Hfus: float, target_Vm: float,
    target_hsp: Optional[HSP],
    cand_MW: float, cand_Tm: float,
    cand_Hfus: float, cand_Vm: float,
    cand_hsp: Optional[HSP],
    w_MW: float = 1.0,
    w_Tm: float = 1.0,
    w_Hfus: float = 1.0,
    w_Vm: float = 1.0,
    w_dD: float = 1.0,
    w_dP: float = 1.0,
    w_dH: float = 1.0,
) -> float:
    """
    Weighted normalised Euclidean distance in
    (MW, Tm, Hfus, Vm, dD, dP, dH) space.

    MW, Tm, Hfus, Vm use fractional differences.
    HSP components use absolute differences scaled by 15 MPa^0.5.
    """
    # Fractional deviations
    d_MW   = ((cand_MW   - target_MW)   / max(target_MW,   1.0)) ** 2
    d_Tm   = ((cand_Tm   - target_Tm)   / max(target_Tm,   1.0)) ** 2
    d_Hfus = ((cand_Hfus - target_Hfus) / max(abs(target_Hfus), 0.1)) ** 2
    d_Vm   = ((cand_Vm   - target_Vm)   / max(target_Vm,   1.0)) ** 2

    # HSP: absolute difference / typical magnitude
    if target_hsp and cand_hsp:
        scale = 15.0
        d_dD = ((cand_hsp.dD - target_hsp.dD) / scale) ** 2
        d_dP = ((cand_hsp.dP - target_hsp.dP) / scale) ** 2
        d_dH = ((cand_hsp.dH - target_hsp.dH) / scale) ** 2
    elif target_hsp and not cand_hsp:
        # Penalty: prefer candidates that have HSP data
        d_dD = d_dP = d_dH = 0.25
    else:
        # No target HSP → match on MW + Tm + Hfus + Vm only
        d_dD = d_dP = d_dH = 0.0
        w_MW *= 1.5
        w_Tm *= 1.5

    return np.sqrt(
        w_MW * d_MW + w_Tm * d_Tm +
        w_Hfus * d_Hfus + w_Vm * d_Vm +
        w_dD * d_dD + w_dP * d_dP + w_dH * d_dH
    )


def find_proxy(
    compound: Union[str, dict, CompoundInput],
    target_smiles: Optional[str] = None,
    pool: Optional[List[ProxyCandidate]] = None,
    n_top: int = 5,
    weights: Optional[Dict[str, float]] = None,
    verbose: bool = False,
) -> ProxyMatch:
    """
    Find the best thermosteam-compatible proxy for a target compound.

    Parameters
    ----------
    compound : str, dict, or CompoundInput
        Chemical identifier.  Can be a plain name string, or a dict /
        CompoundInput with optional 'cas', 'chebi', and 'smiles' fields::

            'Ethanol'
            {'name': 'Syringaldehyde', 'cas': '134-96-3'}
            {'name': 'Cyanidin', 'chebi': 'CHEBI:80159', 'smiles': '...'}

    target_smiles : str, optional
        Explicit SMILES override (kept for backward compatibility;
        prefer putting smiles in the compound dict).
    pool : list[ProxyCandidate], optional
        Pre-built proxy pool.  Built automatically if not supplied.
    n_top : int
        Number of top candidates to keep in the result.
    weights : dict, optional
        Override distance weights.  Keys: 'MW', 'Tb', 'dD', 'dP', 'dH'.
        Default is 1.0 for each.

    Returns
    -------
    ProxyMatch with the best proxy and diagnostics.
    """
    # Normalise input
    ci = _normalise_compound(compound)
    target_name = ci.name

    # Merge SMILES from dict and legacy argument (dict takes priority)
    if ci.smiles:
        target_smiles = ci.smiles
    cas = ci.cas
    chebi = ci.chebi

    # ── 0. Resolve ChEBI → name / CAS / SMILES via PubChem ─────────
    #    (enriches identifiers before the main lookup chain)
    chebi_resolved = {}
    if chebi:
        chebi_resolved = _resolve_chebi_to_identifiers(chebi)
        # Fill in any missing identifiers from the resolved data
        if not cas and chebi_resolved.get('cas'):
            cas = chebi_resolved['cas']
        if not target_smiles and chebi_resolved.get('smiles'):
            target_smiles = chebi_resolved['smiles']

    w = {'MW': 1.0, 'Tm': 1.0, 'Hfus': 1.0, 'Vm': 1.0,
         'dD': 1.0, 'dP': 1.0, 'dH': 1.0}
    if weights:
        w.update(weights)

    # ── 1. Check if thermosteam already has it ──────────────────────
    #    Try: name → CAS → name resolved from ChEBI
    identifiers_to_try = [target_name, cas]
    if chebi_resolved.get('name'):
        identifiers_to_try.append(chebi_resolved['name'])
    if chebi_resolved.get('cas') and chebi_resolved['cas'] != cas:
        identifiers_to_try.append(chebi_resolved['cas'])

    for identifier in identifiers_to_try:
        if identifier is None:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                chem = tmo.Chemical(identifier)
                # Reject if Hf is missing — 0*NaN poisons Hnet in HXutility
                if chem.Hf is None or (isinstance(chem.Hf, float) and math.isnan(chem.Hf)):
                    continue

                # ── Gas-phase enthalpy check ───────────────────────
                # Only needed if the compound has meaningful vapor
                # pressure.  Compounds with Psat < 1e-5 Pa at 100°C
                # will be phase-locked to liquid downstream, so
                # gas-phase data is irrelevant — accept them as-is.
                _needs_gas_check = True
                if not getattr(chem, 'locked_state', None) and chem.Tb is not None:
                    try:
                        _psat = chem.Psat(373.15)
                    except Exception:
                        _psat = 0.0
                    if _psat < 1e-5:
                        _needs_gas_check = False  # will be phase-locked

                if _needs_gas_check and not getattr(chem, 'locked_state', None) and chem.Tb is not None:
                    _gas_ok = True
                    for _T in [chem.Tb + 10, chem.Tb + 50, chem.Tb + 100]:
                        try:
                            _hval = chem.H('g', _T, 101325)
                            if _hval is None:
                                _gas_ok = False
                                break
                        except Exception:
                            _gas_ok = False
                            break
                    if not _gas_ok:
                        if verbose:
                            print(f"  ⚠  '{target_name}' loaded in thermosteam but "
                                  f"H('g') fails near Tb → searching for proxy")
                        continue  # fall through to proxy search

                return ProxyMatch(
                    target_name=target_name,
                    target_smiles=target_smiles,
                    target_MW=chem.MW,
                    target_Tm=None, target_Hfus=None,
                    target_Vm=None,
                    target_hsp=None,
                    in_thermosteam=True,
                    thermosteam_id=chem.ID,
                )
        except Exception:
            continue

    # ── 2. Get target SMILES ────────────────────────────────────────
    #
    # Full lookup chain, tried in order until one succeeds:
    #   a. User-supplied SMILES (from dict or smiles_map)
    #   b. Check if target_name is itself a valid SMILES string
    #   c. chemicals library by name (local PubChem index, ~5200)
    #   d. chemicals library by CAS
    #   e. lookup_smiles(name) from Polymersolubility.py:
    #        → BiopolymerSMILESLookup (PubChem + ChEBI OLS)
    #        → PubChem REST API (by name)
    #        → NIH CIR (by name)
    #   f. PubChem REST API by CAS number
    #   g. PubChem REST API by ChEBI identifier
    #   h. lookup_smiles(CAS) — PubChem/CIR with the CAS string
    #
    smiles = target_smiles
    if ci.smiles:
        smiles_source = 'user_supplied'
    elif smiles and chebi_resolved.get('smiles'):
        smiles_source = 'ChEBI_resolved'
    else:
        smiles_source = None

    # (b) Name is itself a SMILES?
    if smiles is None and is_valid_smiles(target_name):
        smiles = target_name
        smiles_source = 'direct_SMILES'

    # (c) chemicals library by name
    if smiles is None:
        try:
            from chemicals.identifiers import search_chemical
            info = search_chemical(target_name)
            if info and info.smiles:
                smiles = info.smiles
                smiles_source = 'chemicals_library'
        except (ValueError, Exception):
            pass  # Not in local library — continue to network lookups

    # (d) chemicals library by CAS
    if smiles is None and cas:
        try:
            from chemicals.identifiers import search_chemical
            info = search_chemical(cas)
            if info and info.smiles:
                smiles = info.smiles
                smiles_source = 'chemicals_library_CAS'
        except (ValueError, Exception):
            pass  # Not in local library — continue to network lookups

    # (e) Polymersolubility lookup chain by name
    #     (PubChem REST, ChEBI OLS, NIH CIR)
    if smiles is None:
        try:
            found, src = lookup_smiles(target_name)
            if found and is_valid_smiles(found):
                smiles = found
                smiles_source = src
        except Exception as e:
            if verbose:
                print(f"  [DEBUG] (e) lookup_smiles('{target_name}') failed: {type(e).__name__}: {e}")

    # (f) PubChem REST API by CAS
    if smiles is None and cas:
        try:
            found = _lookup_smiles_by_cas(cas)
            if found and is_valid_smiles(found):
                smiles = found
                smiles_source = 'PubChem_REST_CAS'
        except Exception as e:
            if verbose:
                print(f"  [DEBUG] (f) _lookup_smiles_by_cas('{cas}') failed: {type(e).__name__}: {e}")

    # (g) PubChem REST API by ChEBI
    if smiles is None and chebi:
        try:
            found = _lookup_smiles_by_chebi(chebi)
            if found and is_valid_smiles(found):
                smiles = found
                smiles_source = 'PubChem_REST_ChEBI'
        except Exception as e:
            if verbose:
                print(f"  [DEBUG] (g) _lookup_smiles_by_chebi('{chebi}') failed: {type(e).__name__}: {e}")

    # (h) Polymersolubility lookup chain with CAS as search term
    if smiles is None and cas:
        try:
            found, src = lookup_smiles(cas)
            if found and is_valid_smiles(found):
                smiles = found
                smiles_source = f'{src}_via_CAS'
        except Exception as e:
            if verbose:
                print(f"  [DEBUG] (h) lookup_smiles('{cas}') failed: {type(e).__name__}: {e}")

    if smiles is None:
        if verbose:
            print(f"  [DEBUG] All SMILES lookups exhausted for '{target_name}' — no proxy possible")
        return ProxyMatch(
            target_name=target_name, target_smiles=None,
            target_MW=None, target_Tm=None, target_Hfus=None,
            target_Vm=None, target_hsp=None,
            in_thermosteam=False, smiles_source=None,
        )

    # ── 3. Estimate target properties ───────────────────────────────
    #
    #   All from Polymersolubility.py / GROUP_CONTRIBUTIONS:
    #     HSP  → estimate_hsp_from_smiles()     (Hoftyzer-Van Krevelen)
    #     MW   → RDKit Descriptors.MolWt()
    #     Tm   → estimate_tm_from_smiles()      (Van Krevelen Table 6.8)
    #     Hfus → estimate_hfus_from_smiles()    (Van Krevelen Table 5.7)
    #     Vm   → estimate_vm_from_smiles()      (Van Krevelen Table 4.10)
    #
    target_hsp  = estimate_hsp_from_smiles(smiles)
    target_MW   = get_mw_from_smiles(smiles)
    target_Tm   = estimate_tm_from_smiles(smiles)
    target_Hfus = estimate_hfus_from_smiles(smiles)
    target_Vm   = estimate_vm_from_smiles(smiles)

    if target_MW is None:
        return ProxyMatch(
            target_name=target_name, target_smiles=smiles,
            target_MW=None, target_Tm=None, target_Hfus=None,
            target_Vm=None, target_hsp=target_hsp,
            in_thermosteam=False, smiles_source=smiles_source,
        )

    # ── 4. Screen the pool ──────────────────────────────────────────
    if pool is None:
        pool = build_proxy_pool(verbose=True)

    scored = []
    for cand in pool:
        dist = _normalised_distance(
            target_MW, target_Tm or 300,
            target_Hfus or 5.0, target_Vm or 100,
            target_hsp,
            cand.MW, cand.Tm or 300,
            cand.Hfus or 5.0, cand.Vm or 100,
            cand.hsp,
            w_MW=w['MW'], w_Tm=w['Tm'],
            w_Hfus=w['Hfus'], w_Vm=w['Vm'],
            w_dD=w['dD'], w_dP=w['dP'], w_dH=w['dH'],
        )
        scored.append((dist, cand))

    scored.sort(key=lambda x: x[0])
    top = scored[:n_top]

    if not top:
        return ProxyMatch(
            target_name=target_name, target_smiles=smiles,
            target_MW=target_MW, target_Tm=target_Tm,
            target_Hfus=target_Hfus, target_Vm=target_Vm,
            target_hsp=target_hsp, in_thermosteam=False,
            smiles_source=smiles_source,
        )

    best_dist, best = top[0]

    return ProxyMatch(
        target_name=target_name, target_smiles=smiles,
        target_MW=target_MW, target_Tm=target_Tm,
        target_Hfus=target_Hfus, target_Vm=target_Vm,
        target_hsp=target_hsp, in_thermosteam=False,
        smiles_source=smiles_source,
        proxy_name=best.name, proxy_cas=best.cas,
        proxy_MW=best.MW, proxy_Tm=best.Tm,
        proxy_Hfus=best.Hfus, proxy_Vm=best.Vm,
        proxy_hsp=best.hsp,
        distance=best_dist,
        top_candidates=[
            {
                'name': c.name, 'cas': c.cas,
                'MW': c.MW, 'Tm': c.Tm, 'Hfus': c.Hfus, 'Vm': c.Vm,
                'hsp': {'dD': c.hsp.dD, 'dP': c.hsp.dP, 'dH': c.hsp.dH}
                    if c.hsp else None,
                'distance': round(d, 5),
            }
            for d, c in top
        ],
    )


# ═══════════════════════════════════════════════════════════════════════
# SERIALISATION HELPERS  (for assign_proxies result caching)
# ═══════════════════════════════════════════════════════════════════════

def _hsp_to_dict(hsp: Optional[HSP]) -> Optional[dict]:
    if hsp is None:
        return None
    return {'dD': hsp.dD, 'dP': hsp.dP, 'dH': hsp.dH, 'source': hsp.source}


def _dict_to_hsp(d: Optional[dict]) -> Optional[HSP]:
    if d is None:
        return None
    return HSP(**d)


def _proxy_match_to_dict(m: ProxyMatch) -> dict:
    return {
        'target_name': m.target_name,
        'target_smiles': m.target_smiles,
        'target_MW': m.target_MW,
        'target_Tm': m.target_Tm,
        'target_Hfus': m.target_Hfus,
        'target_Vm': m.target_Vm,
        'target_hsp': _hsp_to_dict(m.target_hsp),
        'in_thermosteam': m.in_thermosteam,
        'thermosteam_id': m.thermosteam_id,
        'smiles_source': m.smiles_source,
        'proxy_name': m.proxy_name,
        'proxy_cas': m.proxy_cas,
        'proxy_MW': m.proxy_MW,
        'proxy_Tm': m.proxy_Tm,
        'proxy_Hfus': m.proxy_Hfus,
        'proxy_Vm': m.proxy_Vm,
        'proxy_hsp': _hsp_to_dict(m.proxy_hsp),
        'distance': m.distance,
        'top_candidates': m.top_candidates,
    }


def _dict_to_proxy_match(d: dict) -> ProxyMatch:
    d = dict(d)  # shallow copy
    d['target_hsp'] = _dict_to_hsp(d.get('target_hsp'))
    d['proxy_hsp'] = _dict_to_hsp(d.get('proxy_hsp'))
    return ProxyMatch(**d)


def _make_assign_cache_key(
    compounds: list,
    smiles_map: Optional[dict],
    skip: Optional[list],
    weights: Optional[dict],
) -> str:
    """
    Deterministic hash of the inputs to assign_proxies.

    Any change in compound list, order, identifiers, skip list,
    weights, or smiles overrides produces a different key.
    """
    payload = json.dumps(
        {
            'compounds': [
                c if isinstance(c, (str, dict))
                else {'name': c.name, 'cas': c.cas,
                      'chebi': c.chebi, 'smiles': c.smiles}
                for c in compounds
            ],
            'smiles_map': smiles_map or {},
            'skip': sorted(skip) if skip else [],
            'weights': weights or {},
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _load_assign_cache(
    cache_key: str, verbose: bool = True,
) -> Optional[Tuple[Dict[str, ProxyMatch], Dict[str, str]]]:
    """
    Try to load a cached assign_proxies result.

    Returns (report, name_map) on hit, or None on miss.
    The *chemicals* list is NOT cached because thermosteam Chemical
    objects are not serialisable — it is rebuilt cheaply from the
    report (native → tmo.Chemical(id), proxy → tmo.Chemical(proxy).copy()).
    """
    cache_file = os.path.join(ASSIGN_CACHE_DIR, f'{cache_key}.json')
    if not os.path.exists(cache_file):
        return None
    try:
        with open(cache_file) as f:
            data = json.load(f)
        if data.get('version') != 2:
            return None
        report = {
            name: _dict_to_proxy_match(d)
            for name, d in data['report'].items()
        }
        name_map = data['name_map']
        if verbose:
            print(f"[assign-cache] Loaded cached results for {len(report)} "
                  f"compounds (key {cache_key})")
        return report, name_map
    except Exception:
        return None


def _save_assign_cache(
    cache_key: str,
    report: Dict[str, ProxyMatch],
    name_map: Dict[str, str],
    verbose: bool = True,
) -> None:
    """Save assign_proxies results to disk."""
    try:
        os.makedirs(ASSIGN_CACHE_DIR, exist_ok=True)
        cache_file = os.path.join(ASSIGN_CACHE_DIR, f'{cache_key}.json')
        data = {
            'version': 2,
            'report': {
                name: _proxy_match_to_dict(m)
                for name, m in report.items()
            },
            'name_map': name_map,
        }
        with open(cache_file, 'w') as f:
            json.dump(data, f)
        if verbose:
            print(f"[assign-cache] Saved results → {cache_file}")
    except Exception as e:
        if verbose:
            print(f"[assign-cache] Could not save cache: {e}")


# ═══════════════════════════════════════════════════════════════════════
# BATCH PROCESSING + THERMOSTEAM INTEGRATION
# ═══════════════════════════════════════════════════════════════════════

def _backfill_solid_volume(chem, vm_cm3: Optional[float] = None,
                           verbose: bool = False) -> None:
    """
    Ensure *chem* has a solid molar volume method (V.s).

    Many thermosteam chemicals — especially proxies — only carry liquid-
    phase property correlations.  When a BioSTEAM unit (e.g. StorageTank)
    tries to compute the volumetric flow of a solid-phase stream it calls
    ``chem.V.s(T)``, which raises if no method is registered.

    This function tests for that and, when the method is missing, adds a
    constant-value fallback:

    1. Use the Van Krevelen group-contribution Vm already estimated by the
       proxy finder (passed as *vm_cm3*, in cm³ mol⁻¹).
    2. If that is unavailable, estimate from MW assuming a typical organic-
       solid density of 1 200 kg m⁻³.

    Parameters
    ----------
    chem : thermosteam.Chemical
        The chemical object to patch (modified in place).
    vm_cm3 : float, optional
        Van Krevelen molar volume estimate in cm³ mol⁻¹.
    verbose : bool
        Print a message when a fallback is added.
    """
    # Quick check — does V.s already work?
    try:
        chem.V.s(298.15)
        return                       # nothing to do
    except Exception:
        pass

    # Build the best constant estimate we can
    if vm_cm3 and vm_cm3 > 0:
        Vm_m3 = vm_cm3 * 1e-6       # cm³ mol⁻¹  →  m³ mol⁻¹
        source = 'Van Krevelen Vm'
    else:
        # Fallback: assume solid density ≈ 1 200 kg m⁻³
        Vm_m3 = chem.MW / (1200.0 * 1e3)   # m³ mol⁻¹
        source = 'MW / 1200 kg m⁻³'

    # thermosteam expects V.s(T) → m³ mol⁻¹
    # Strategy 1: V.s is a proper TDependentProperty with add_method
    Vs = getattr(chem.V, 's', None)
    if Vs is not None and hasattr(Vs, 'add_method'):
        Vs.add_method(lambda T, _Vm=Vm_m3: _Vm)
        if verbose:
            print(f"  ⚠  Backfilled V.s for '{chem.ID}' "
                  f"({source}, Vm = {Vm_m3:.3e} m³/mol)")
        return

    # Strategy 2: V.s is missing or is a placeholder (e.g. a string).
    # Create a fresh VolumeSolid object and wire it into the phase handle.
    try:
        from thermo import VolumeSolid
        vs_obj = VolumeSolid(CASRN=getattr(chem, 'CAS', ''),
                             MW=chem.MW)
        vs_obj.add_method(lambda T, _Vm=Vm_m3: _Vm)
        chem.V.s = vs_obj
        if verbose:
            print(f"  ⚠  Backfilled V.s (new obj) for '{chem.ID}' "
                  f"({source}, Vm = {Vm_m3:.3e} m³/mol)")
        return
    except Exception:
        pass

    # Strategy 3: Monkey-patch V.s as a simple callable so that
    # downstream code calling chem.V.s(T) gets a float back.
    try:
        chem.V.s = lambda T, _Vm=Vm_m3: _Vm
        if verbose:
            print(f"  ⚠  Backfilled V.s (callable) for '{chem.ID}' "
                  f"({source}, Vm = {Vm_m3:.3e} m³/mol)")
        return
    except Exception:
        pass

    if verbose:
        print(f"  ⚠  Could not backfill V.s for '{chem.ID}' "
              f"— StorageTank may fail on solid-phase streams")


def _rebuild_chemicals_from_report(
    report: Dict[str, ProxyMatch],
    name_map: Dict[str, str],
    verbose: bool = True,
) -> list:
    """
    Rebuild the chemicals list from a cached report.

    This is the only work needed on a cache hit — fast local
    tmo.Chemical() calls, no SMILES lookups or API calls.
    """
    chemicals: list = []
    for name, match in report.items():
        if match.smiles_source == 'skipped':
            continue

        if match.in_thermosteam:
            ts_id = match.thermosteam_id
            if ts_id is not None:
                chemicals.append(ts_id)
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    try:
                        chem_obj = tmo.Chemical(name)
                    except Exception:
                        chem_obj = None
                    if chem_obj is not None:
                        _backfill_solid_volume(chem_obj, verbose=verbose)
                        chemicals.append(chem_obj)
        elif match.proxy_name:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                try:
                    proxy_chem = tmo.Chemical(match.proxy_name)
                except Exception:
                    proxy_chem = tmo.Chemical(match.proxy_cas)
                copy = proxy_chem.copy(name)
                _backfill_solid_volume(
                    copy,
                    vm_cm3=match.target_Vm or match.proxy_Vm,
                    verbose=verbose,
                )
                chemicals.append(copy)
        # else: no proxy found — skip, same as original logic

    return chemicals


def assign_proxies(
    compounds: List[Union[str, dict, CompoundInput]],
    smiles_map: Optional[Dict[str, str]] = None,
    skip: Optional[List[str]] = None,
    weights: Optional[Dict[str, float]] = None,
    verbose: bool = True,
    use_cache: bool = True,
) -> Tuple[List, Dict[str, 'ProxyMatch'], Dict[str, str]]:
    """
    For a list of compounds, check thermosteam availability and
    assign proxies for those that are missing.

    Results are cached to disk so that repeated calls with the same
    compound list skip all SMILES lookups, API calls, and proxy
    matching.  Set *use_cache=False* to force a fresh run.

    Parameters
    ----------
    compounds : list
        Chemical specifications.  Each entry can be:
          - a plain string:  'Ethanol'
          - a dict:  {'name': 'X', 'cas': '...', 'chebi': '...', 'smiles': '...'}
          - a CompoundInput instance
        Only 'name' is required in the dict; cas/chebi/smiles are optional
        and provide extra lookup paths.
    smiles_map : dict, optional
        {name: SMILES} overrides (legacy interface; prefer putting
        smiles in the compound dicts).
    skip : list[str], optional
        Compound names to skip entirely.  Use this for compounds that
        are already defined as custom chemicals in your BioSTEAM model
        (e.g. Cellulose, Hemicellulose, Lignin, starch) which are
        created with search_db=False and won't be found by
        tmo.Chemical() with default database search.  Skipped compounds
        appear in the report as 'skip' status and are NOT included in
        the returned chemicals list (since you handle them yourself).
    weights : dict, optional
        Distance weights – see find_proxy().
    verbose : bool
        Print progress and summary table.
    use_cache : bool
        If True (default), check for a cached result before doing
        the full resolution.  Set False to force a fresh run.

    Returns
    -------
    chemicals : list[str | thermosteam.Chemical]
        Ready-to-use items for tmo.Chemicals().  Native compounds are
        returned as plain strings (their thermosteam ID) so that
        tmo.Chemicals() builds them with full phase-locks and property
        guards.  Proxy compounds are returned as Chemical objects with
        the proxy's thermo data and the user's original name as the ID.
        Does NOT include skipped compounds.
    report : dict[str, ProxyMatch]
        Detailed match info keyed by the user's original compound name.
    name_map : dict[str, str]
        Mapping of {thermosteam_id: original_user_name} for every
        compound whose thermosteam name differs from the user's name.
        Use this to translate stream compositions back to user names
        on output.  Compounds where the names match are omitted.
    """
    if smiles_map is None:
        smiles_map = {}
    skip_set = set(s.lower().strip() for s in (skip or []))

    # ── Try loading from cache ──────────────────────────────────────
    cache_key = _make_assign_cache_key(compounds, smiles_map, skip, weights)

    if use_cache:
        cached = _load_assign_cache(cache_key, verbose=verbose)
        if cached is not None:
            report, name_map = cached
            chemicals = _rebuild_chemicals_from_report(
                report, name_map, verbose=verbose,
            )
            if verbose:
                n_native = sum(1 for m in report.values() if m.in_thermosteam)
                n_proxy = sum(1 for m in report.values()
                              if not m.in_thermosteam and m.proxy_name)
                n_fail = sum(1 for m in report.values()
                             if not m.in_thermosteam and m.proxy_name is None)
                print(f" Summary (cached): {n_native} native  |  "
                      f"{n_proxy} proxied  |  {n_fail} failed")
            return chemicals, report, name_map

    # ── Full resolution (no cache hit) ──────────────────────────────
    # Build pool once
    pool = build_proxy_pool(verbose=verbose)

    report: Dict[str, ProxyMatch] = {}
    chemicals: list = []                  # mix of str (native) and Chemical (proxy)
    name_map: Dict[str, str] = {}         # {thermosteam_id: original_user_name}

    if verbose:
        print(f"\n{'='*72}")
        print(f" Processing {len(compounds)} compounds")
        print(f"{'='*72}")

    for entry in compounds:
        ci = _normalise_compound(entry)

        # Skip compounds already defined in the user's BioSTEAM model
        if ci.name.lower().strip() in skip_set:
            match = ProxyMatch(
                target_name=ci.name, target_smiles=None,
                target_MW=None, target_Tm=None, target_Hfus=None,
                target_Vm=None, target_hsp=None,
                in_thermosteam=True,  # treat as available
                smiles_source='skipped',
            )
            report[ci.name] = match
            if verbose:
                print(f"  ⊘  {ci.name:30s}"
                      f" — skipped (user-defined in BioSTEAM model)")
            continue

        # Merge legacy smiles_map
        if ci.name in smiles_map and not ci.smiles:
            ci.smiles = smiles_map[ci.name]

        match = find_proxy(ci, pool=pool, weights=weights, verbose=verbose)
        report[ci.name] = match

        if match.in_thermosteam:
            # ── Native compound: pass as a STRING so tmo.Chemicals()
            #    builds it from its own database with full phase-locks
            #    and property guards (e.g. solid-only for Cellulose).
            ts_id = match.thermosteam_id   # the ID that actually loaded
            if ts_id is not None:
                chemicals.append(ts_id)     # plain string → native build
                # Record mapping if the thermosteam name differs from user name
                if ts_id != ci.name:
                    name_map[ts_id] = ci.name
                    if verbose:
                        print(f"  ℹ  '{ci.name}' → thermosteam ID '{ts_id}'")
            else:
                # Fallback: shouldn't happen, but try the old path
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    try:
                        chem_obj = tmo.Chemical(ci.name)
                    except Exception:
                        if ci.cas:
                            _tmp = tmo.Chemical(ci.cas)
                            chem_obj = _tmp.copy(ci.name)
                        else:
                            chem_obj = None
                    if chem_obj is not None:
                        _backfill_solid_volume(chem_obj, verbose=verbose)
                        chemicals.append(chem_obj)
        elif match.proxy_name:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                try:
                    proxy_chem = tmo.Chemical(match.proxy_name)
                except Exception:
                    proxy_chem = tmo.Chemical(match.proxy_cas)
                copy = proxy_chem.copy(ci.name)
                # Use target Vm first (better match to actual compound),
                # fall back to proxy Vm
                _backfill_solid_volume(
                    copy,
                    vm_cm3=match.target_Vm or match.proxy_Vm,
                    verbose=verbose,
                )
                chemicals.append(copy)
        else:
            if verbose:
                print(f"  WARNING: No proxy found for '{ci.name}' – skipped")

        if verbose:
            print(match.summary())

    if verbose:
        n_native = sum(1 for m in report.values() if m.in_thermosteam)
        n_proxy = sum(1 for m in report.values()
                      if not m.in_thermosteam and m.proxy_name)
        n_fail = sum(1 for m in report.values()
                     if not m.in_thermosteam and m.proxy_name is None)
        print(f"\n{'─'*72}")
        print(f" Summary: {n_native} native  |  {n_proxy} proxied"
              f"  |  {n_fail} failed")
        if name_map:
            print(f" Name mappings (thermosteam_id → original):")
            for ts_id, orig in name_map.items():
                print(f"   {ts_id:30s} → {orig}")
        print(f"{'─'*72}\n")

    # ── Save to cache ───────────────────────────────────────────────
    if use_cache:
        _save_assign_cache(cache_key, report, name_map, verbose=verbose)

    return chemicals, report, name_map


# ═══════════════════════════════════════════════════════════════════════
# CONVENIENCE: build a complete thermosteam Thermo object
# ═══════════════════════════════════════════════════════════════════════

def create_thermo(
    compounds: List[Union[str, dict, CompoundInput]],
    smiles_map: Optional[Dict[str, str]] = None,
    weights: Optional[Dict[str, float]] = None,
    verbose: bool = True,
) -> Tuple[tmo.Thermo, Dict[str, 'ProxyMatch'], Dict[str, str]]:
    """
    One-call setup: returns a tmo.Thermo with all compounds
    (native or proxied), plus a name mapping.

    Usage
    -----
    >>> thermo, report, name_map = create_thermo([
    ...     'Water',
    ...     'Ethanol',
    ...     {'name': 'Syringaldehyde', 'cas': '134-96-3'},
    ...     {'name': 'Cyanidin-3-glucoside',
    ...      'chebi': 'CHEBI:80159',
    ...      'smiles': 'OC1C(OC2=CC3=...'},
    ... ])
    >>> tmo.settings.set_thermo(thermo)
    """
    chems, report, name_map = assign_proxies(
        compounds, smiles_map=smiles_map, weights=weights, verbose=verbose,
    )
    chemicals_obj = tmo.Chemicals(chems)
    thermo = tmo.Thermo(chemicals_obj)
    return thermo, report, name_map
