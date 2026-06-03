"""
Flory-Huggins Polymer Solubility Model

Predicts equilibrium polymer concentration from polymer and solvent

All parameters derived from name:
- HSP: Database lookup OR Hoftyzer-Van Krevelen (1976) estimation from SMILES
- Density: Database lookup OR Van Krevelen (2009) Table 4.10 group contribution
- N: Database lookup OR default median of biopolymer literature (N=368)

References:

Theoretical Basis:
- Flory, P.J. (1953). Principles of Polymer Chemistry. Cornell University Press.
- Lindvig, T. et al. (2002). Fluid Phase Equilib. 203, 247-260.
  DOI:10.1016/S0378-3812(02)00184-X (Modified Flory-Huggins with HSP)

Solubility Parameters:
- Hansen, C.M. (2007). Hansen Solubility Parameters: A User's Handbook, 2nd Ed.
  CRC Press. (Tables A.1 for solvents, A.2 for polymers)
- Hoftyzer, P.J. & Van Krevelen, D.W. (1976). Properties of Polymers, 2nd Ed.
  (HSP group contribution method, Table 7.10 in 4th Ed.)
- Lee, H.L. et al. (1991). J Adhesion Sci Technol 5(5), 377-396.
  DOI:10.1080/02773819108050276 (Cellulose HSP)
- Sameni, J. et al. (2017). BioResources 12(1), 1548-1565.
  DOI:10.15376/biores.12.1.1548-1565 (Lignin HSP)

Polymer Physical Properties:
- Van Krevelen, D.W. & te Nijenhuis, K. (2009). Properties of Polymers, 4th Ed.
  Elsevier. Ch.4: Density
- Van Krevelen, D.W. (1990). Properties of Polymers, 3rd Ed. Elsevier.
  (Earlier edition also referenced in database)
- Brandrup, J., Immergut, E.H., Grulke, E.A. (1999). Polymer Handbook, 4th Ed.
  Wiley. (Polymer physical properties database)
- Mark, J.E. (1999). Polymer Data Handbook. Oxford University Press.

Plant Polymer Data:
- Ebringerova, A. et al. (2005). Adv Polym Sci 186, 1-67.
  DOI:10.1007/b136816 (Hemicellulose structure and properties)
- Lindman, B. et al. (2010). Langmuir 26(8), 5251-5259. (Cellulose solubility)
- Medronho, B. et al. (2012). Cellulose 19, 581-587. (Cellulose dissolution)
"""
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, List
import urllib.request
import urllib.parse
import json

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Fragments
    # Suppress RDKit warnings and errors from being printed to console
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

# Try to import the robust biopolymer SMILES lookup module
try:
    from biopolymer_smiles import BiopolymerSMILESLookup, get_biopolymer_smiles
    BIOPOLYMER_SMILES_AVAILABLE = True
except ImportError:
    BIOPOLYMER_SMILES_AVAILABLE = False

# Constants
R = 8.314  # J/(mol·K)
T_REF = 298.15  # K


def lookup_smiles_from_pubchem(name: str) -> Optional[str]:
    """
    Look up SMILES from PubChem using REST API.

    Tries multiple search strategies:
    1. Direct name lookup
    2. With "poly" removed if present (search for monomer)
    3. Common name variations

    Returns canonical SMILES or None if not found.
    """
    # Clean up the name
    name = name.strip().lower()

    # Generate search variants
    search_names = [name]

    # If starts with "poly", also try the monomer name
    if name.startswith('poly'):
        monomer = name[4:]  # Remove "poly"
        # Handle poly(X) format
        if monomer.startswith('(') and monomer.endswith(')'):
            monomer = monomer[1:-1]
        search_names.append(monomer)

        # Common variations
        if monomer.endswith('ene'):
            search_names.append(monomer)  # ethylene, propylene, etc.
        if monomer.endswith('ic acid'):
            # lactic acid -> lactide
            search_names.append(monomer.replace('ic acid', 'ide'))
        if monomer.endswith('amide'):
            search_names.append(monomer)

    # Remove common polymer prefixes/suffixes for search
    clean_name = name.replace('polymer', '').replace('poly ', '').strip()
    if clean_name and clean_name != name:
        search_names.append(clean_name)

    for search_name in search_names:
        if not search_name:
            continue
        try:
            # URL encode the name
            encoded_name = urllib.parse.quote(search_name)
            url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded_name}/property/CanonicalSMILES/JSON"

            req = urllib.request.Request(url, headers={'User-Agent': 'Python/PolymerSolubility'})
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode())

                if 'PropertyTable' in data and 'Properties' in data['PropertyTable']:
                    smiles = data['PropertyTable']['Properties'][0].get('CanonicalSMILES')
                    if smiles:
                        return smiles
        except:
            continue

    return None


def lookup_smiles_from_cir(name: str) -> Optional[str]:
    """
    Look up SMILES from NIH Chemical Identifier Resolver (CIR).

    CIR is a backup service that can resolve many chemical names to SMILES.

    Returns SMILES or None if not found.
    """
    # Clean up the name
    name = name.strip()

    # Generate search variants
    search_names = [name]

    if name.lower().startswith('poly'):
        monomer = name[4:].strip()
        if monomer.startswith('(') and monomer.endswith(')'):
            monomer = monomer[1:-1]
        search_names.append(monomer)

    for search_name in search_names:
        if not search_name:
            continue
        try:
            encoded_name = urllib.parse.quote(search_name)
            url = f"https://cactus.nci.nih.gov/chemical/structure/{encoded_name}/smiles"

            req = urllib.request.Request(url, headers={'User-Agent': 'Python/PolymerSolubility'})
            with urllib.request.urlopen(req, timeout=10) as response:
                smiles = response.read().decode().strip()
                if smiles and not smiles.startswith('<!'):  # Not an HTML error page
                    return smiles
        except:
            continue

    return None


def lookup_smiles(name: str) -> Tuple[Optional[str], str]:
    """
    Look up SMILES from multiple external sources in order of preference.

    Search order:
    1. BiopolymerSMILESLookup (robust PubChem + ChEBI via OLS API) - PREFERRED
    2. PubChem REST API (fallback)
    3. NIH CIR (last resort backup)

    Returns (SMILES, source) or (None, 'not_found')
    """
    # 1. Try the robust BiopolymerSMILESLookup module first (PubChem + ChEBI OLS)
    if BIOPOLYMER_SMILES_AVAILABLE:
        try:
            lookup = BiopolymerSMILESLookup(verbose=False)
            result = lookup.lookup(name)
            if result["monomers"]:
                monomer = result["monomers"][0]
                smiles = monomer.get("smiles")
                source = monomer.get("source", "BiopolymerSMILES")
                if smiles:
                    if "pubchem" in source.lower():
                        return smiles, 'BiopolymerSMILES_PubChem'
                    elif "chebi" in source.lower():
                        return smiles, 'BiopolymerSMILES_ChEBI'
                    else:
                        return smiles, f'BiopolymerSMILES_{source}'
        except Exception:
            pass  # Fall through to fallback methods

    # 2. Fallback: Try PubChem REST API
    pubchem = lookup_smiles_from_pubchem(name)
    if pubchem:
        return pubchem, 'PubChem_REST'

    # 3. Last resort: Try NIH CIR as backup
    cir = lookup_smiles_from_cir(name)
    if cir:
        return cir, 'NIH_CIR'

    return None, 'not_found'




def _looks_like_smiles(text: str) -> bool:
    """
    Quick heuristic check if text could plausibly be a SMILES string.

    SMILES strings typically contain special characters like =, #, (, ), [, ], @, etc.
    Plain chemical names typically only contain letters, numbers, spaces, and hyphens.

    This avoids passing obvious non-SMILES strings to RDKit which generates parse errors.
    """
    # If it contains typical SMILES special characters, it's likely SMILES
    smiles_special_chars = set('=()[]#@+\\/.')
    if any(c in text for c in smiles_special_chars):
        return True

    # Check for common chemical name patterns that indicate it's NOT a SMILES
    text_lower = text.lower()
    name_indicators = [
        'poly', 'acid', 'amine', 'ether', 'ester', 'alcohol', 'aldehyde',
        'ketone', 'cellulose', 'lignin', 'nylon', 'vinyl', 'styrene',
        'ethyl', 'methyl', 'propyl', 'butyl', 'phenyl', 'benzyl',
        'oxide', 'sulfide', 'chloride', 'bromide', 'fluoride',
        ' '  # spaces indicate a name
    ]
    if any(indicator in text_lower for indicator in name_indicators):
        return False

    # Valid single-letter organic SMILES elements (uppercase)
    single_letter_elements = set('CNOSPFIHB')

    # All uppercase short strings need special handling
    if text.isupper() and len(text) <= 6:
        # If composed entirely of valid SMILES single-letter elements, it could be SMILES
        # e.g., "CCO" (ethanol), "CCCC" (butane), "CO" (formaldehyde)
        if all(c in single_letter_elements for c in text):
            return True
        # Otherwise it's likely an abbreviation like PMMA, PVC, PET
        return False

    # Check if it's composed only of valid SMILES atom characters (including lowercase aromatic)
    valid_smiles_chars = set('CNOSPFIHBcnospfihb')
    clean = text.replace(' ', '')
    if all(c in valid_smiles_chars or c.isdigit() for c in clean):
        return True

    return False


def is_valid_smiles(text: str) -> bool:
    """Check if text is a valid SMILES string."""
    if not RDKIT_AVAILABLE:
        return False

    # Quick check - if it doesn't look like SMILES, don't bother parsing
    if not _looks_like_smiles(text):
        return False

    try:
        mol = Chem.MolFromSmiles(text)
        return mol is not None
    except:
        return False


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class HSP:
    dD: float
    dP: float
    dH: float
    source: str = ''

    @property
    def total(self):
        return np.sqrt(self.dD**2 + self.dP**2 + self.dH**2)


@dataclass
class SolventData:
    hsp: HSP
    Vm: float
    source: str


@dataclass
class PolymerData:
    """All polymer data needed for Flory-Huggins"""
    hsp: HSP
    hsp_source: str
    density: float  # g/cm³
    density_source: str
    N: float = 368  # Degree of polymerization (default: median of biopolymer literature values)
    N_source: str = 'default_median_biopolymer'


# =============================================================================
# SOLVENT DATABASE - Hansen (2007) Table A.1
# =============================================================================

SOLVENT_DB: Dict[str, SolventData] = {
    'water': SolventData(HSP(15.5, 16.0, 42.3), 18.0, 'Hansen2007_A1'),
    'methanol': SolventData(HSP(15.1, 12.3, 22.3), 40.7, 'Hansen2007_A1'),
    'ethanol': SolventData(HSP(15.8, 8.8, 19.4), 58.5, 'Hansen2007_A1'),
    '1-propanol': SolventData(HSP(16.0, 6.8, 17.4), 75.2, 'Hansen2007_A1'),
    '2-propanol': SolventData(HSP(15.8, 6.1, 16.4), 76.8, 'Hansen2007_A1'),
    '1-butanol': SolventData(HSP(16.0, 5.7, 15.8), 91.5, 'Hansen2007_A1'),
    'ethylene glycol': SolventData(HSP(17.0, 11.0, 26.0), 55.8, 'Hansen2007_A1'),
    'glycerol': SolventData(HSP(17.4, 12.1, 29.3), 73.3, 'Hansen2007_A1'),
    'acetone': SolventData(HSP(15.5, 10.4, 7.0), 74.0, 'Hansen2007_A1'),
    'methyl ethyl ketone': SolventData(HSP(16.0, 9.0, 5.1), 90.1, 'Hansen2007_A1'),
    'cyclohexanone': SolventData(HSP(17.8, 6.3, 5.1), 104.0, 'Hansen2007_A1'),
    'dmso': SolventData(HSP(18.4, 16.4, 10.2), 71.3, 'Hansen2007_A1'),
    'dmf': SolventData(HSP(17.4, 13.7, 11.3), 77.0, 'Hansen2007_A1'),
    'nmp': SolventData(HSP(18.0, 12.3, 7.2), 96.5, 'Hansen2007_A1'),
    'acetonitrile': SolventData(HSP(15.3, 18.0, 6.1), 52.9, 'Hansen2007_A1'),
    'thf': SolventData(HSP(16.8, 5.7, 8.0), 81.7, 'Hansen2007_A1'),
    'diethyl ether': SolventData(HSP(14.5, 2.9, 5.1), 104.8, 'Hansen2007_A1'),
    '1,4-dioxane': SolventData(HSP(19.0, 1.8, 7.4), 85.7, 'Hansen2007_A1'),
    'ethyl acetate': SolventData(HSP(15.8, 5.3, 7.2), 98.5, 'Hansen2007_A1'),
    'butyl acetate': SolventData(HSP(15.8, 3.7, 6.3), 132.5, 'Hansen2007_A1'),
    'chloroform': SolventData(HSP(17.8, 3.1, 5.7), 80.7, 'Hansen2007_A1'),
    'dichloromethane': SolventData(HSP(18.2, 6.3, 6.1), 64.0, 'Hansen2007_A1'),
    'carbon tetrachloride': SolventData(HSP(17.8, 0.0, 0.6), 97.1, 'Hansen2007_A1'),
    'benzene': SolventData(HSP(18.4, 0.0, 2.0), 89.4, 'Hansen2007_A1'),
    'toluene': SolventData(HSP(18.0, 1.4, 2.0), 106.8, 'Hansen2007_A1'),
    'xylene': SolventData(HSP(17.6, 1.0, 3.1), 123.3, 'Hansen2007_A1'),
    'hexane': SolventData(HSP(14.9, 0.0, 0.0), 131.6, 'Hansen2007_A1'),
    'heptane': SolventData(HSP(15.3, 0.0, 0.0), 147.4, 'Hansen2007_A1'),
    'cyclohexane': SolventData(HSP(16.8, 0.0, 0.2), 108.7, 'Hansen2007_A1'),
}


# =============================================================================
# POLYMER DATABASE - Complete data for known polymers
# HSP: Hansen (2007) Table A.2, Lee (1991), Sameni (2017)
# Physical: Van Krevelen (1990), Brandrup (1999), Mark (1999)
# =============================================================================

POLYMER_DB: Dict[str, PolymerData] = {
    'cellulose': PolymerData(
        hsp=HSP(25.4, 18.6, 24.8), hsp_source='Hansen2007_A2',
        density=1.56, density_source='Mark1999',
        N=11000, N_source='DOI:10.15586/qas.v17i4.1581_mean_7000-15000',
    ),
    'cellulose acetate': PolymerData(
        hsp=HSP(14.9, 7.1, 11.1), hsp_source='Hansen2007_A2',
        density=1.30, density_source='Mark1999',
    ),
    'xylan': PolymerData(
        hsp=HSP(20.1, 13.2, 22.4), hsp_source='estimated_HoftyzerVanKrevelen',
        density=1.52, density_source='VanKrevelen2009',
        N=100, N_source='DOI:10.3390/molecules25010135_mean_50-200',
    ),
    'mannans': PolymerData(
        hsp=HSP(20.0, 13.0, 22.0), hsp_source='estimated_HoftyzerVanKrevelen',
        density=1.52, density_source='VanKrevelen2009',
        N=65, N_source='DOI:10.3390/molecules25010135_mean_60-70',
    ),
    'starch': PolymerData(
        hsp=HSP(19.6, 14.7, 22.4), hsp_source='estimated_HoftyzerVanKrevelen',
        density=1.50, density_source='Mark1999',
        N=68, N_source='DOI:10.3390/polym14112215_mean_15-120',
    ),
    'pectin': PolymerData(
        hsp=HSP(20.3, 11.4, 21.2), hsp_source='estimated_HoftyzerVanKrevelen',
        density=1.52, density_source='VanKrevelen2009',
        N=12, N_source='DOI:10.1016/0308-8146(94)90088-4_mean_7-18',
    ),
    'agarose': PolymerData(
        hsp=HSP(19.8, 13.5, 23.0), hsp_source='estimated_HoftyzerVanKrevelen',
        density=1.50, density_source='estimated',
        N=800, N_source='ISBN_978-0879691363',
    ),
    'carrageenan': PolymerData(
        hsp=HSP(19.5, 14.0, 23.5), hsp_source='estimated_HoftyzerVanKrevelen',
        density=1.50, density_source='estimated',
        N=650, N_source='DOI:10.3390/polysaccharides3030037_mean_300-1000',
    ),
    'dextran': PolymerData(
        hsp=HSP(20.0, 14.2, 23.0), hsp_source='estimated_HoftyzerVanKrevelen',
        density=1.50, density_source='estimated',
        N=610, N_source='DOI:10.3390/macromol5030034_mean_20-1200',
    ),
    'pullulan': PolymerData(
        hsp=HSP(20.0, 14.0, 22.8), hsp_source='estimated_HoftyzerVanKrevelen',
        density=1.50, density_source='estimated',
        N=645, N_source='DOI:10.3390/polysaccharides3030037_mean_90-1200',
    ),
    'lignin': PolymerData(
        hsp=HSP(20.17, 14.61, 15.04), hsp_source='Hansen2007_A2',
        density=1.30, density_source='sigma-aldirch',
        N=100, N_source='Sameni2017_Mw~9000_M0~180',
    ),
}

POLYMER_ALIASES = {
    'mannan': 'mannans',
    'galactomannan': 'mannans',
    'hemicellulose': 'xylan',
    'kappa carrageenan': 'carrageenan',
    'iota carrageenan': 'carrageenan',
    'lambda carrageenan': 'carrageenan',
    'kraft lignin': 'lignin',
}


# =============================================================================
# SMILES-BASED ESTIMATION - Consolidated Group Contributions
# =============================================================================

# Unified group contribution data from Van Krevelen (2009)
# Keys: group name
# Values: dict with properties:
#   - Fd: Dispersion attraction constant (MJ/m³)^0.5·mol⁻¹ for HSP (Table 7.10)
#   - Fp: Polar component (MJ/m³)^0.5·mol⁻¹ for HSP (Table 7.10)
#   - Eh: H-bonding energy J/mol for HSP (Table 7.10)
#   - V: Molar volume cm³/mol for density (Table 4.10)
#   - Ym: Melt transition function K·g/mol for Tm (Table 6.8)
#   - Hm: Heat of fusion kJ/mol for Hfus (Table 5.7)

GROUP_CONTRIBUTIONS = {
    # Aliphatic carbons
    'CH3':      {'Fd': 420,  'Fp': 0,    'Eh': 0,     'V': 16.5,  'Ym': 4000,  'Hm': 2.0},
    'CH2':      {'Fd': 270,  'Fp': 0,    'Eh': 0,     'V': 16.37, 'Ym': 5700,  'Hm': 4.0},
    'CH':       {'Fd': 80,   'Fp': 0,    'Eh': 0,     'V': 12.0,  'Ym': 7000,  'Hm': 2.0},
    'C':        {'Fd': -70,  'Fp': 0,    'Eh': 0,     'V': 8.0,   'Ym': 5000,  'Hm': 1.0},

    # Substituted carbons
    'CH(CH3)':  {'Fd': 350,  'Fp': 0,    'Eh': 0,     'V': 32.72, 'Ym': 13000, 'Hm': 4.7},
    'C(CH3)2':  {'Fd': 630,  'Fp': 0,    'Eh': 0,     'V': 49.0,  'Ym': 18000, 'Hm': 8.6},
    'CH(C6H5)': {'Fd': 1510, 'Fp': 110,  'Eh': 0,     'V': 77.5,  'Ym': 48000, 'Hm': 6.0},

    # Unsaturated
    'CH2_db':   {'Fd': 400,  'Fp': 0,    'Eh': 0,     'V': 14.0,  'Ym': 5000,  'Hm': 2.0},
    'CH_db':    {'Fd': 200,  'Fp': 0,    'Eh': 0,     'V': 10.0,  'Ym': 4750,  'Hm': 0.25},
    'C_db':     {'Fd': 70,   'Fp': 0,    'Eh': 0,     'V': 6.0,   'Ym': 4000,  'Hm': 0.0},
    'CH=CH':    {'Fd': 400,  'Fp': 0,    'Eh': 0,     'V': 27.0,  'Ym': 9500,  'Hm': 0.5},
    'C#C':      {'Fd': 400,  'Fp': 0,    'Eh': 0,     'V': 25.0,  'Ym': 8000,  'Hm': 0.0},

    # Aromatic
    'phenyl':   {'Fd': 1430, 'Fp': 110,  'Eh': 0,     'V': 65.5,  'Ym': 45000, 'Hm': 5.0},

    # Halogens
    'F':        {'Fd': 220,  'Fp': 0,    'Eh': 0,     'V': 8.0,   'Ym': 8000,  'Hm': 1.0},
    'Cl':       {'Fd': 450,  'Fp': 550,  'Eh': 400,   'V': 18.0,  'Ym': 10000, 'Hm': 2.0},
    'Br':       {'Fd': 550,  'Fp': 0,    'Eh': 0,     'V': 22.0,  'Ym': 12000, 'Hm': 2.5},
    'CHF':      {'Fd': 300,  'Fp': 0,    'Eh': 0,     'V': 20.0,  'Ym': 38000, 'Hm': 3.5},
    'CHCl':     {'Fd': 530,  'Fp': 550,  'Eh': 400,   'V': 30.0,  'Ym': 17400, 'Hm': 7.0},
    'CF2':      {'Fd': 440,  'Fp': 0,    'Eh': 0,     'V': 23.7,  'Ym': 25500, 'Hm': 4.0},
    'CCl2':     {'Fd': 900,  'Fp': 1100, 'Eh': 800,   'V': 40.1,  'Ym': 39000, 'Hm': 4.0},
    'CFCl':     {'Fd': 670,  'Fp': 550,  'Eh': 400,   'V': 32.0,  'Ym': 32000, 'Hm': 2.0},

    # Oxygen groups
    'OH':       {'Fd': 210,  'Fp': 500,  'Eh': 20000, 'V': 12.5,  'Ym': 15000, 'Hm': 2.0},
    'O':        {'Fd': 100,  'Fp': 400,  'Eh': 3000,  'V': 8.5,   'Ym': 13500, 'Hm': 1.0},
    'CHO':      {'Fd': 470,  'Fp': 800,  'Eh': 4500,  'V': 18.0,  'Ym': 20000, 'Hm': 1.5},
    'CO':       {'Fd': 290,  'Fp': 770,  'Eh': 2000,  'V': 13.5,  'Ym': 28000, 'Hm': 0.0},
    'COOH':     {'Fd': 530,  'Fp': 420,  'Eh': 10000, 'V': 28.0,  'Ym': 35000, 'Hm': 3.0},
    'COO':      {'Fd': 390,  'Fp': 490,  'Eh': 7000,  'V': 23.0,  'Ym': 28000, 'Hm': -2.5},
    'CONH':     {'Fd': 450,  'Fp': 770,  'Eh': 11000, 'V': 21.0,  'Ym': 50000, 'Hm': 2.0},

    # Nitrogen groups
    'NH2':      {'Fd': 280,  'Fp': 0,    'Eh': 8400,  'V': 14.0,  'Ym': 20000, 'Hm': 2.0},
    'NH':       {'Fd': 160,  'Fp': 210,  'Eh': 3100,  'V': 6.4,   'Ym': 18000, 'Hm': 1.5},
    'N':        {'Fd': 20,   'Fp': 800,  'Eh': 5000,  'V': 7.0,   'Ym': 15000, 'Hm': 1.0},
    'CN':       {'Fd': 430,  'Fp': 1100, 'Eh': 2500,  'V': 22.0,  'Ym': 25000, 'Hm': 2.0},
    'NO2':      {'Fd': 500,  'Fp': 1070, 'Eh': 1500,  'V': 28.0,  'Ym': 30000, 'Hm': 2.5},

    # Sulfur groups
    'S':        {'Fd': 440,  'Fp': 0,    'Eh': 0,     'V': 17.3,  'Ym': 22500, 'Hm': -1.5},

    # Ring contribution (for aliphatic rings)
    'ring':     {'Fd': 190,  'Fp': 0,    'Eh': 0,     'V': 0,     'Ym': 0,     'Hm': 0},
}

# SMARTS patterns for group matching, ordered from most specific to least specific
# Format: (SMARTS, group_name)
SMARTS_PATTERNS = [
    # Aromatic rings (check first)
    ('c1ccccc1', 'phenyl'),

    # Substituted carbons (check before simple carbons)
    ('[CH;X4](c1ccccc1)', 'CH(C6H5)'),
    ('[C;X4]([CH3])([CH3])', 'C(CH3)2'),
    ('[CH;X4]([CH3])', 'CH(CH3)'),

    # Carbonyl groups (check before simple O)
    ('C(=O)N', 'CONH'),
    ('C(=O)O', 'COO'),
    ('C(=O)[OH]', 'COOH'),
    ('[CH]=O', 'CHO'),
    ('C(=O)', 'CO'),

    # Nitrogen groups
    ('[NH2]', 'NH2'),
    ('[NH;X3]', 'NH'),
    ('[N;X3]', 'N'),
    ('C#N', 'CN'),
    ('[N+](=O)[O-]', 'NO2'),

    # Hydroxyl and ether
    ('[OH]', 'OH'),
    ('[O;X2]', 'O'),

    # Sulfur
    ('[S;X2]', 'S'),

    # Halogenated carbons (check before simple halogens)
    ('[C;X4]([F])([Cl])', 'CFCl'),
    ('[C;X4]([F])([F])', 'CF2'),
    ('[C;X4]([Cl])([Cl])', 'CCl2'),
    ('[CH;X4]([F])', 'CHF'),
    ('[CH;X4]([Cl])', 'CHCl'),

    # Simple halogens
    ('[F]', 'F'),
    ('[Cl]', 'Cl'),
    ('[Br]', 'Br'),

    # Unsaturated carbons
    ('[CH]=[CH]', 'CH=CH'),
    ('C#C', 'C#C'),
    ('[CH2]=[C]', 'CH2_db'),
    ('[CH]=[C]', 'CH_db'),
    ('[C]=[C]', 'C_db'),

    # Simple aliphatic carbons (check last)
    ('[CH3]', 'CH3'),
    ('[CH2;X4]', 'CH2'),
    ('[CH;X4]', 'CH'),
    ('[C;X4]', 'C'),
]


def _match_smarts_groups(smiles: str) -> Optional[Dict[str, int]]:
    """
    Match SMARTS patterns against a SMILES string and return group counts.

    Uses the unified SMARTS_PATTERNS list, matching most specific patterns first
    and avoiding double-counting atoms.

    Returns dict mapping group names to counts, or None if SMILES is invalid.
    """
    if not RDKIT_AVAILABLE:
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mol_h = Chem.AddHs(mol)
    assigned_atoms = set()
    group_counts = {}

    for smarts, group_name in SMARTS_PATTERNS:
        try:
            pattern = Chem.MolFromSmarts(smarts)
            if pattern is None:
                continue
            matches = mol_h.GetSubstructMatches(pattern)
            for match in matches:
                # Skip if any atom already assigned
                if any(idx in assigned_atoms for idx in match):
                    continue
                # Count this group and mark atoms
                group_counts[group_name] = group_counts.get(group_name, 0) + 1
                assigned_atoms.update(match)
        except:
            continue

    # Count aliphatic rings (non-aromatic)
    ring_info = mol.GetRingInfo()
    if ring_info.NumRings() > 0:
        n_aromatic_rings = sum(1 for ring in ring_info.AtomRings()
                               if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring))
        n_aliphatic_rings = ring_info.NumRings() - n_aromatic_rings
        if n_aliphatic_rings > 0:
            group_counts['ring'] = n_aliphatic_rings

    return group_counts


def estimate_hsp_from_smiles(smiles: str) -> Optional[HSP]:
    """
    Hoftyzer-Van Krevelen (1976) group contribution for polymer HSP.

    Reference: Van Krevelen (2009), Chapter 7, Table 7.10

    Equations:
        δd = ΣFdi / V
        δp = √(ΣFpi²) / V
        δh = √(ΣEhi / V)
    """
    groups = _match_smarts_groups(smiles)
    if groups is None:
        return None

    sum_Fd = 0.0
    sum_Fp2 = 0.0
    sum_Eh = 0.0
    sum_V = 0.0

    for group_name, count in groups.items():
        if group_name in GROUP_CONTRIBUTIONS:
            g = GROUP_CONTRIBUTIONS[group_name]
            sum_Fd += g['Fd'] * count
            sum_Fp2 += (g['Fp'] ** 2) * count
            sum_Eh += g['Eh'] * count
            sum_V += g['V'] * count

    # Fallback if volume too small
    if sum_V < 30:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            sum_V = Descriptors.MolWt(mol) * 0.95

    if sum_V <= 0:
        return None

    # Calculate HSP components
    delta_d = sum_Fd / sum_V
    delta_p = np.sqrt(sum_Fp2) / sum_V
    delta_h = np.sqrt(sum_Eh / sum_V) if sum_Eh > 0 else 0.0

    # Apply bounds
    dD = max(13.0, min(25.0, round(delta_d, 1)))
    dP = max(0.0, min(18.0, round(delta_p, 1)))
    dH = max(0.0, min(25.0, round(delta_h, 1)))

    return HSP(dD, dP, dH, 'estimated_HoftyzerVanKrevelen_Table7.10')


def estimate_density_from_smiles(smiles: str) -> Optional[Tuple[float, str]]:
    """
    Van Krevelen (2009) molar volume group contribution for density.

    Reference: Properties of Polymers, 4th Ed., Chapter 4, Table 4.10

    ρ = MW / Vm
    """
    if not RDKIT_AVAILABLE:
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    groups = _match_smarts_groups(smiles)
    if groups is None:
        return None

    MW = Descriptors.MolWt(mol)
    Vm = 0.0

    for group_name, count in groups.items():
        if group_name in GROUP_CONTRIBUTIONS:
            Vm += GROUP_CONTRIBUTIONS[group_name]['V'] * count

    # Fallback if couldn't match much
    if Vm < 20:
        Vm = MW * 0.95

    density = MW / Vm if Vm > 0 else 1.0
    density = max(0.85, min(2.3, density))

    return round(density, 2), 'estimated_VanKrevelen2009_Table4.10'



# =============================================================================
# GET POLYMER DATA
# =============================================================================

def get_polymer_data(polymer_input: str) -> Tuple[Optional[PolymerData], str]:
    """
    Get all polymer data from database or by looking up SMILES and estimating properties.

    Search order:
    1. Database lookup (known polymers with literature values)
    2. SMILES lookup (PubChem → NIH CIR) + property estimation
    3. Direct SMILES input + property estimation

    Returns (PolymerData, method_description)
    """
    normalized = polymer_input.lower().strip().replace('-', ' ').replace('_', ' ')

    # Check aliases
    if normalized in POLYMER_ALIASES:
        key = POLYMER_ALIASES[normalized]
        if key in POLYMER_DB:
            return POLYMER_DB[key], f'database_alias:{key}'

    # Direct lookup
    if normalized in POLYMER_DB:
        return POLYMER_DB[normalized], 'database'

    # Not in database - try to find SMILES and estimate properties
    if RDKIT_AVAILABLE:
        smiles_to_use = None
        smiles_source = None

        # Check if input is already a valid SMILES
        if is_valid_smiles(polymer_input):
            smiles_to_use = polymer_input
            smiles_source = 'direct_SMILES'
        else:
            # Look up SMILES from various sources (PubChem, NIH CIR)
            smiles, source = lookup_smiles(polymer_input)
            if smiles and is_valid_smiles(smiles):
                smiles_to_use = smiles
                smiles_source = f'{source}:{polymer_input}'

        if smiles_to_use:
            hsp = estimate_hsp_from_smiles(smiles_to_use)
            density_result = estimate_density_from_smiles(smiles_to_use)

            if hsp is not None and density_result is not None:
                density, density_source = density_result

                return PolymerData(
                    hsp=hsp,
                    hsp_source=f"{hsp.source} (SMILES from {smiles_source})",
                    density=density,
                    density_source=f"{density_source} (SMILES from {smiles_source})",
                    N=368,  # Default: median of biopolymer literature values
                    N_source='default_median_biopolymer',
                ), f'estimated_via_{smiles_source}'

    return None, 'not_found'


# =============================================================================
# FLORY-HUGGINS CALCULATIONS
# =============================================================================

def _select_alpha(s_hsp: HSP) -> float:
    """
    Select the correction constant α based on solvent HSP class.

    Classification follows Lindvig et al. (2002) Table 3 optimal α values:
        - H-bonding + polar (δhb > 13, δp > 8):  α = 0.80 (average of polar and H-bonding optima)
        - H-bonding only    (δhb > 13, δp ≤ 8):  α = 0.60 (H-bonding optimum)
        - Polar only        (δhb ≤ 13, δp > 8):  α = 1.00 (polar optimum)
        - Non-polar         (δhb ≤ 13, δp ≤ 8):  α = 0.55 (non-polar optimum)

    Thresholds derived from HSP classification boundaries in Lindvig Tables 6-9.
    """
    if s_hsp.dH > 13 and s_hsp.dP > 8:
        return 0.80  # H-bonding + polar (e.g. water, ethylene glycol)
    elif s_hsp.dH > 13:
        return 0.60  # H-bonding only (e.g. 1-butanol, 1-propanol, cyclohexanol)
    elif s_hsp.dP > 8:
        return 1.00  # Polar (e.g. acetone, DMSO, DMF, acetonitrile)
    else:
        return 0.55  # Non-polar (e.g. hexane, toluene, chloroform)


def calculate_chi(p_hsp: HSP, s_hsp: HSP, Vm: float, T: float = T_REF, alpha: float = None) -> float:
    """
    χ₁₂ = (α·V₁/RT) × [(δ₁d−δ₂d)² + 0.25(δ₁p−δ₂p)² + 0.25(δ₁hb−δ₂hb)²]

    α is automatically selected based on solvent HSP class (Lindvig et al. 2002,
    Table 3) unless explicitly provided.

    Reference: Lindvig et al. (2002) DOI:10.1016/S0378-3812(02)00184-X
    """
    if alpha is None:
        alpha = _select_alpha(s_hsp)
    dD = p_hsp.dD - s_hsp.dD
    dP = p_hsp.dP - s_hsp.dP
    dH = p_hsp.dH - s_hsp.dH
    return (alpha * Vm * (dD**2 + 0.25 * dP**2 + 0.25 * dH**2)) / (R * T)



def calculate_chi_critical(N: float) -> float:
    """
    Critical Flory-Huggins interaction parameter.

    For polymers (N > 1):
        χ_c = 0.5 × (1 + 1/√N)²

    For small molecules (N = 1):
        χ_c = 0.5  (symmetric mixture limit)

    References
    ----------
    Flory, P.J. (1953). Principles of Polymer Chemistry. Cornell UP.
    Qian, D. et al. (2022). J. Phys. Chem. Lett. 13, 7853-7860, eq 3.
        DOI:10.1021/acs.jpclett.2c01986
    """
    if N <= 1:
        return 0.5
    return 0.5 * (1 + 1/np.sqrt(N))**2


def find_equilibrium_phi(chi: float, N: float) -> Tuple[float, str]:
    """
    Find equilibrium polymer volume fraction in dilute phase using
    a crossover solubility equation that interpolates between the
    mean-field and scaling regimes.

    For chi <= chi_c: system is fully miscible (phi = 1).
    For chi > chi_c:  dilute-phase binodal volume fraction is computed
    from the crossover formula:

        phi^- = [1 - exp(num_arg)] / [1 - exp(den_arg)]

    where num_arg and den_arg are defined through auxiliary variables
    alpha, Delta, A, B, and D (see below).

    Auxiliary variables (computed from N and chi):
        alpha = N^(1/4)                           # scaling
        chi_c = 0.5 * (1 + 1/sqrt(N))^2           # criticality
        Delta = (chi - chi_c) / chi_c              # reduced distance
        A = (1/alpha) * (1 + Delta/alpha^2) * sqrt(3*Delta)         # stepping
        B = alpha * (1 + alpha^2 * Delta) * sqrt(3*Delta)             # stepping
        D = cosh(ln(alpha)) / (coth(A) + coth(B))               # invariance

    References
    ----------
    Flory, P.J. (1953). Principles of Polymer Chemistry. Cornell UP.
    Qian, D., Michaels, T.C.T. & Knowles, T.P.J. (2022).
        J. Phys. Chem. Lett. 13(33), 7853-7860.
        DOI:10.1021/acs.jpclett.2c01986
    """
    chi_c = calculate_chi_critical(N)

    if chi <= chi_c:
        return 1.0, 'miscible'

    # --- Auxiliary variables ---
    alpha = N ** 0.25                                        # Scaling
    Delta = (chi - chi_c) / chi_c                            # Reduced distance

    ln_alpha = np.log(alpha)

    # Stepping parameters A and B
    sqrt_3Delta = np.sqrt(3.0 * Delta)
    A = (1.0 / alpha) * (1.0 + Delta / alpha**2) * sqrt_3Delta
    B = alpha * (1.0 + alpha**2 * Delta) * sqrt_3Delta

    # Safe coth helper: coth(x) = cosh(x)/sinh(x)
    # For large |x|, coth(x) -> sign(x) * 1.0
    def _coth(x):
        if abs(x) < 1e-12:
            return np.sign(x) * 1e12                         # coth -> +/-inf near 0
        if abs(x) > 500:
            return np.sign(x) * 1.0                          # coth -> +/-1 for large x
        return np.cosh(x) / np.sinh(x)

    coth_A = _coth(A)
    coth_B = _coth(B)

    # Invariance parameter D
    D = np.cosh(ln_alpha) / (coth_A + coth_B)

    # --- Crossover solubility equation ---
    alpha2   = alpha ** 2         # alpha^2
    alpha_m2 = alpha ** (-2)      # alpha^(-2)
    sinh_lna = np.sinh(ln_alpha)

    # Numerator exponent:  +8D[sinh(ln alpha)/alpha^2 + D(1+Delta)(coth B / alpha^2)]
    num_exp = 8.0 * D * (
        alpha_m2 * sinh_lna
        + D * (1.0 + Delta) * (alpha_m2 * coth_B)
    )

    # Denominator exponent: +8D[(1/alpha^2 - alpha^2) sinh(ln alpha)
    #                           + D(1+Delta)(coth B / alpha^2 + alpha^2 coth A)]
    den_exp = 8.0 * D * (
        (alpha_m2 - alpha2) * sinh_lna
        + D * (1.0 + Delta) * (alpha_m2 * coth_B + alpha2 * coth_A)
    )

    # Clamp to prevent overflow in exp()
    num_exp = max(-700.0, min(700.0, num_exp))
    den_exp = max(-700.0, min(700.0, den_exp))

    numerator   = 1.0 - np.exp(num_exp)
    denominator = 1.0 - np.exp(den_exp)

    # Guard against zero denominator
    if abs(denominator) < 1e-30:
        phi = 0.0
    else:
        phi = numerator / denominator

    # Clamp to physical range [0, 1]
    phi = max(0.0, min(1.0, phi))

    # Below 1e-10 is effectively insoluble
    if phi < 1e-10:
        phi = 0.0
        return phi, 'immiscible'

    # Classify
    if phi > 0.1:
        status = 'partial'
    elif phi > 1e-4:
        status = 'low_solubility'
    else:
        status = 'immiscible'

    return phi, status


# =============================================================================
# MAIN PREDICTION FUNCTION
# =============================================================================

def predict_solubility(polymer_name: str, solvent_name: str, T: float = T_REF) -> dict:
    """
    Predict equilibrium polymer concentration from names only.

    Parameters
    ----------
    polymer_name : str
        Polymer name (e.g., 'polystyrene') OR repeat unit SMILES
    solvent_name : str
        Solvent name (e.g., 'toluene')
    T : float
        Temperature (K), default 298.15

    Returns
    -------
    dict with:
        - solubility_g_L: Equilibrium concentration (g/L)
        - phi_polymer: Volume fraction in dilute phase
        - chi: Flory-Huggins parameter
        - status: miscible/partial/low_solubility/immiscible
        - All source information for traceability
    """
    # Get solvent
    s_key = solvent_name.lower().strip()
    if s_key not in SOLVENT_DB:
        return {'error': f'Solvent "{solvent_name}" not found'}
    solvent = SOLVENT_DB[s_key]

    # Get polymer data
    polymer, method = get_polymer_data(polymer_name)
    if polymer is None:
        return {'error': f'Cannot get data for "{polymer_name}". '
                f'RDKit available: {RDKIT_AVAILABLE}'}

    # Calculate chi
    chi_hsp = calculate_chi(polymer.hsp, solvent.hsp, solvent.Vm, T)
    chi_total = chi_hsp
    chi_c = calculate_chi_critical(polymer.N)

    # Find equilibrium
    phi, status = find_equilibrium_phi(chi_total, polymer.N)

    # Convert to g/L
    solubility = phi * polymer.density * 1000

    # Ra (Hansen distance)
    Ra = np.sqrt(
        4*(polymer.hsp.dD - solvent.hsp.dD)**2 +
        (polymer.hsp.dP - solvent.hsp.dP)**2 +
        (polymer.hsp.dH - solvent.hsp.dH)**2
    )

    return {
        'polymer': polymer_name,
        'solvent': solvent_name,
        'T': T,
        # Main results
        'solubility_g_L': solubility,
        'phi_polymer': phi,
        'chi': chi_total,
        'chi_hsp': chi_hsp,
        'chi_critical': chi_c,
        'Ra': Ra,
        # Polymer data used
        'polymer_hsp': {'dD': polymer.hsp.dD, 'dP': polymer.hsp.dP, 'dH': polymer.hsp.dH},
        'polymer_hsp_source': polymer.hsp_source,
        'density': polymer.density,
        'density_source': polymer.density_source,
        'N': polymer.N,
        'N_source': polymer.N_source,
        # Solvent data
        'solvent_hsp': {'dD': solvent.hsp.dD, 'dP': solvent.hsp.dP, 'dH': solvent.hsp.dH},
        'solvent_Vm': solvent.Vm,
        'solvent_source': solvent.source,
        # Method
        'polymer_data_method': method,
    }




# =============================================================================
# SMALL MOLECULE / COMPOUND SOLUBILITY PREDICTION
# =============================================================================

def get_compound_data(compound_input: str) -> Tuple[Optional[PolymerData], str]:
    """
    Get data for a small molecule/compound (not a polymer).

    Similar to get_polymer_data but sets N=1 for monomeric compounds.
    This is useful for compounds not in thermosteam that lack UNIFAC groups.

    Parameters
    ----------
    compound_input : str
        Compound name (e.g., 'glucose', 'vanillin') OR SMILES string

    Returns
    -------
    (CompoundData, method_description) or (None, 'not_found')
    """
    # Check if it's in the solvent database first
    normalized = compound_input.lower().strip()
    if normalized in SOLVENT_DB:
        # Convert solvent data to PolymerData format with N=1
        solvent = SOLVENT_DB[normalized]
        # Estimate density from molar volume and typical MW
        estimated_density = 1.0  # Default
        return PolymerData(
            hsp=solvent.hsp,
            hsp_source=f'solvent_database:{solvent.source}',
            density=estimated_density,
            density_source='estimated',
            N=1,  # Monomer
            N_source='small_molecule',
        ), 'solvent_database'

    # Try to find SMILES and estimate properties
    if RDKIT_AVAILABLE:
        smiles_to_use = None
        smiles_source = None

        # Check if input is already a valid SMILES
        if is_valid_smiles(compound_input):
            smiles_to_use = compound_input
            smiles_source = 'direct_SMILES'
        else:
            # Look up SMILES from various sources
            smiles, source = lookup_smiles(compound_input)
            if smiles and is_valid_smiles(smiles):
                smiles_to_use = smiles
                smiles_source = f'{source}:{compound_input}'

        if smiles_to_use:
            hsp = estimate_hsp_from_smiles(smiles_to_use)
            density_result = estimate_density_from_smiles(smiles_to_use)

            if hsp is not None and density_result is not None:
                density, density_source = density_result

                return PolymerData(
                    hsp=hsp,
                    hsp_source=f"{hsp.source} (SMILES from {smiles_source})",
                    density=density,
                    density_source=f"{density_source} (SMILES from {smiles_source})",
                    N=1,  # Small molecule - degree of polymerization = 1
                    N_source='small_molecule',
                ), f'estimated_small_molecule_via_{smiles_source}'

    return None, 'not_found'


def predict_compound_solubility(solute_name: str, solvent_name: str, T: float = T_REF) -> dict:
    """
    Predict solubility of a small molecule compound in a solvent (g/L).

    This is an alternative to UNIFAC-based methods for compounds not in
    thermosteam or lacking suitable UNIFAC groups. Uses Hansen Solubility
    Parameters with Flory-Huggins (N=1).

    Parameters
    ----------
    solute_name : str
        Compound name (e.g., 'glucose', 'vanillin', 'quercetin') OR SMILES
    solvent_name : str
        Solvent name (e.g., 'water', 'ethanol')
    T : float
        Temperature (K), default 298.15

    Returns
    -------
    dict with:
        - solubility_g_L: Equilibrium concentration (g/L)
        - phi_solute: Volume fraction
        - chi: Flory-Huggins parameter
        - status: miscible/partial/low_solubility/immiscible
        - Ra: Hansen distance
    """
    # Get solvent
    s_key = solvent_name.lower().strip()
    if s_key not in SOLVENT_DB:
        return {'error': f'Solvent "{solvent_name}" not found in database'}
    solvent = SOLVENT_DB[s_key]

    # Get solute data
    solute, method = get_compound_data(solute_name)
    if solute is None:
        return {'error': f'Cannot get data for "{solute_name}". '
                f'RDKit available: {RDKIT_AVAILABLE}'}

    # Calculate chi (N=1 for small molecules)
    chi_hsp = calculate_chi(solute.hsp, solvent.hsp, solvent.Vm, T)
    chi_total = chi_hsp
    chi_c = calculate_chi_critical(solute.N)  # For N=1, chi_c = 2.0

    # Find equilibrium volume fraction
    phi, _ = find_equilibrium_phi(chi_total, solute.N)

    # Convert to g/L
    solubility = phi * solute.density * 1000

    # Ra (Hansen distance)
    Ra = np.sqrt(
        4*(solute.hsp.dD - solvent.hsp.dD)**2 +
        (solute.hsp.dP - solvent.hsp.dP)**2 +
        (solute.hsp.dH - solvent.hsp.dH)**2
    )

    return {
        'solute': solute_name,
        'solvent': solvent_name,
        'T': T,
        # Main results
        'solubility_g_L': solubility,
        'phi_solute': phi,
        'chi': chi_total,
        'chi_critical': chi_c,
        'Ra': Ra,
        # Solute data
        'solute_hsp': {'dD': solute.hsp.dD, 'dP': solute.hsp.dP, 'dH': solute.hsp.dH},
        'solute_hsp_source': solute.hsp_source,
        'solute_density': solute.density,
        # Solvent data
        'solvent_hsp': {'dD': solvent.hsp.dD, 'dP': solvent.hsp.dP, 'dH': solvent.hsp.dH},
        'solvent_Vm': solvent.Vm,
        'solvent_source': solvent.source,
        # Method
        'data_method': method,
    }


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("FLORY-HUGGINS SOLUBILITY MODEL - EXAMPLE")
    print("=" * 70)

    # Test biopolymers
    print("\n--- BIOPOLYMER SOLUBILITY (g/L) ---")
    polymers = ['cellulose', 'lignin', 'chitosan']
    solvents = ['water', 'dmso', 'ethanol']

    for polymer in polymers:
        print(f"\n{polymer.upper()}:")
        for solvent in solvents:
            r = predict_solubility(polymer, solvent)
            if 'error' not in r:
                print(f"  {solvent}: {r['solubility_g_L']:.2f} g/L (χ={r['chi']:.3f})")
            else:
                print(f"  {solvent}: {r['error']}")

    # Test small molecules (compounds)
    print("\n--- SMALL MOLECULE SOLUBILITY (g/L) ---")
    compounds = ['glucose', 'Cyanidin-3-rutinoside']
    solvents = ['water', 'ethanol', 'dmso']

    for compound in compounds:
        print(f"\n{compound.upper()}:")
        for solvent in solvents:
            r = predict_compound_solubility(compound, solvent)
            if 'error' not in r:
                print(f"  {solvent}: {r['solubility_g_L']:.2f} g/L (χ={r['chi']:.3f})")
            else:
                print(f"  {solvent}: {r['error']}")