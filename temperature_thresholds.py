"""
temperature_thresholds.py
=========================

Thermal-degradation threshold flagging for the solid-solvent extraction TEA
tools (`extraction_tea_tool.py` and
`simple_extractor_system_for_TEA_tool_with_recycle.py`).

It answers one question for every compound the process actually handles:

    "Was this compound held above the temperature at which it is reported to
     degrade, in the unit that handled it?"

and returns a list of flags that the calling tool prints / displays.

How a compound is matched to a threshold row
--------------------------------------------
1. EXACT MATCH (takes precedence).  Named rows in the table (Vitamin C,
   Lycopene, EGCG, Alpha-Tocopherol, ...) are matched by normalised name,
   CAS number, or ChEBI id.  If a compound matches a named row, that row's
   thresholds are authoritative and group membership is ignored.

2. GROUP MEMBERSHIP via the ChEBI ontology.  For everything else, the
   compound's ChEBI id is walked up the `is_a` hierarchy (using `libchebipy`,
   which downloads + caches the ChEBI database on first use).  If any of a
   group's defining ChEBI classes is an ancestor, the compound belongs to
   that group.  A compound can match several groups (e.g. an anthocyanin is
   also a flavonoid and a glycoside in ChEBI); when it does, the *most
   conservative* (lowest) threshold among the matched groups governs, and the
   output records which group set it.

   A small, clearly-labelled name-substring fallback covers three groups that
   ChEBI does not model as a clean structural class with `is_a` children
   (curcuminoids, capsaicinoids, betalains).

Which threshold column applies to which unit
--------------------------------------------
    Extractor   -> "Heating (Water)"  if the solvent is water,
                   "Heating (Other Solvent)" otherwise
    Evaporator  -> same heating column as the extractor
                   (the table note: "Heating" = extraction and/or solvent
                    evaporation).  Exposure = the HOTTEST effect.
    Dryer       -> "Drying"

All thresholds are in degrees Celsius.  ``None`` means N/A for that medium
(the table lists a non-thermal degradation mechanism, e.g. oxidation- or
hydrolysis-driven), so no flag is raised on temperature for that column.

This module is pure Python (only the optional ChEBI lookup needs a third-party
package).  It deliberately does not import biosteam/thermosteam so it is easy
to test and reuse.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, asdict


# ===========================================================================
# 1.  The threshold table  (°C; None = N/A, no thermal threshold)
# ===========================================================================
# Columns:
#   'water'         -> Heating (Water)          : aqueous extraction / evaporation
#   'other_solvent' -> Heating (Other Solvent)  : non-aqueous extraction / evaporation
#   'drying'        -> Drying
THRESHOLDS = {
    'Vitamin C':              {'water': 90,   'other_solvent': 90,   'drying': 55},
    'Anthocyanins':           {'water': 70,   'other_solvent': 80,   'drying': None},
    'Beta-Carotene':          {'water': 100,  'other_solvent': 100,  'drying': 60},
    'Lycopene':               {'water': 100,  'other_solvent': 100,  'drying': 80},
    'Polyphenols':            {'water': 90,   'other_solvent': 90,   'drying': 100},
    'EGCG':                   {'water': 98,   'other_solvent': 90,   'drying': 100},
    'Allicin':                {'water': 50,   'other_solvent': 50,   'drying': 50},
    'Betalains':              {'water': 50,   'other_solvent': None, 'drying': None},
    'Curcuminoids':           {'water': 90,   'other_solvent': 90,   'drying': 70},
    'Capsaicinoids':          {'water': None, 'other_solvent': None, 'drying': 50},
    'Terpenes':               {'water': 100,  'other_solvent': 100,  'drying': 90},
    'PUFAs':                  {'water': 125,  'other_solvent': 125,  'drying': 125},
    'Alpha-Tocopherol':       {'water': 180,  'other_solvent': 180,  'drying': None},
    'Vitamin B1 (Thiamine)':  {'water': 50,   'other_solvent': 50,   'drying': 50},
    'Phycocyanin':            {'water': 47,   'other_solvent': 47,   'drying': 47},
    'Monoethanolamine (MEA)': {'water': 130,  'other_solvent': 130,  'drying': 130},
    'Glycosides':             {'water': None, 'other_solvent': None, 'drying': 50},
    'Phenolic Compounds':     {'water': 90,   'other_solvent': 90,   'drying': 60},
    'Flavonoids':             {'water': 90,   'other_solvent': 90,   'drying': 60},
    'Levulinic Acid':         {'water': 200,  'other_solvent': 200,  'drying': 200},
}

# Carotenoids (group): any carotenoid that is NOT itself a named row
# (β-carotene and lycopene match exactly and keep their own thresholds) inherits
# the column-wise minimum — i.e. the most conservative — of Beta-Carotene and
# Lycopene. Derived here so it tracks any future edits to those two rows.
THRESHOLDS['Carotenoids'] = {
    col: min(THRESHOLDS['Beta-Carotene'][col], THRESHOLDS['Lycopene'][col])
    for col in ('water', 'other_solvent', 'drying')
}

# 'specific' = matched by exact name/CAS/ChEBI ; 'group' = matched by ChEBI is_a
COMPOUND_TYPE = {
    'Vitamin C': 'specific', 'Beta-Carotene': 'specific', 'Lycopene': 'specific',
    'EGCG': 'specific', 'Allicin': 'specific', 'Alpha-Tocopherol': 'specific',
    'Vitamin B1 (Thiamine)': 'specific', 'Phycocyanin': 'specific',
    'Monoethanolamine (MEA)': 'specific', 'Levulinic Acid': 'specific',
    'Anthocyanins': 'group', 'Polyphenols': 'group', 'Betalains': 'group',
    'Curcuminoids': 'group', 'Capsaicinoids': 'group', 'Terpenes': 'group',
    'PUFAs': 'group', 'Glycosides': 'group', 'Phenolic Compounds': 'group',
    'Flavonoids': 'group',
    'Carotenoids': 'group',
}

COLUMN_LABEL = {
    'water': 'Heating (Water)',
    'other_solvent': 'Heating (Other Solvent)',
    'drying': 'Drying',
}


# ===========================================================================
# 2.  Exact-match specs for the named (specific-compound) rows
# ===========================================================================
# Matched by normalised name OR CAS OR ChEBI id.  Add synonyms freely.
NAMED_COMPOUNDS = {
    'Vitamin C': {
        'names': ['vitamin c', 'ascorbic acid', 'l-ascorbic acid'],
        'cas': ['50-81-7'], 'chebi': [29073, 38290]},
    'Beta-Carotene': {
        'names': ['beta-carotene', 'beta carotene', 'b-carotene'],
        'cas': ['7235-40-7'], 'chebi': [17579]},
    'Lycopene': {
        'names': ['lycopene'], 'cas': ['502-65-8'], 'chebi': [15948]},
    'EGCG': {
        'names': ['egcg', 'epigallocatechin gallate', 'epigallocatechin-3-gallate',
                  'epigallocatechin 3-gallate', '(-)-epigallocatechin gallate'],
        'cas': ['989-51-5'], 'chebi': [4806]},
    'Allicin': {
        'names': ['allicin'], 'cas': ['539-86-6'], 'chebi': [28411]},
    'Alpha-Tocopherol': {
        'names': ['alpha-tocopherol', 'alpha tocopherol', 'a-tocopherol',
                  'd-alpha-tocopherol', '(r,r,r)-alpha-tocopherol',
                  'rrr-alpha-tocopherol', 's,r,r,alpha tocopherol'],
        'cas': ['59-02-9', '10191-41-0', '58-95-7'], 'chebi': [18145]},
    'Vitamin B1 (Thiamine)': {
        'names': ['vitamin b1', 'thiamine', 'thiamin', 'aneurine', 'thiamine(1+)'],
        'cas': ['59-43-8', '70-16-6', '67-03-8'], 'chebi': [18385, 33283, 26948]},
    'Phycocyanin': {
        'names': ['phycocyanin', 'c-phycocyanin'],
        'cas': ['11016-15-2'], 'chebi': []},
    'Monoethanolamine (MEA)': {
        'names': ['monoethanolamine', 'mea', 'ethanolamine', '2-aminoethanol'],
        'cas': ['141-43-5'], 'chebi': [16000]},
    'Levulinic Acid': {
        'names': ['levulinic acid', 'laevulinic acid', 'levulinate',
                  '4-oxopentanoic acid'],
        'cas': ['123-76-2'], 'chebi': [37547]},
}


# ===========================================================================
# 3.  ChEBI structural classes that define each GROUP row
# ===========================================================================
# A compound belongs to a group if ANY of these ChEBI ids is an `is_a`
# ancestor of the compound's ChEBI id.  IDs verified against the live ChEBI
# ontology (May 2026).
#
#   NOTE on Terpenes: the default below includes both `terpene` (35186) and
#   `terpenoid` (26873).  Terpenoid also covers plant sterols / triterpenoids
#   (sitosterol, squalene, amyrins, ...), so those WILL be flagged as terpenes.
#   This is the conservative choice.  To restrict to true terpenes only,
#   change the set to {35186}.
GROUP_CLASSES = {
    'Flavonoids':         {47916},          # flavonoid
    'Polyphenols':        {26195},          # polyphenol
    'Phenolic Compounds': {33853},          # phenols
    'Glycosides':         {24400},          # glycoside
    'Terpenes':           {35186, 26873},   # terpene + terpenoid
    'PUFAs':              {26208},          # polyunsaturated fatty acid
    'Anthocyanins':       {35218, 16366, 38695, 38697},  # anthocyanin/-idin cations
    'Carotenoids':        {23044, 15407},   # carotenoid + carotene
    # Groups ChEBI does not expose as a single is_a class with members:
    'Curcuminoids':       set(),
    'Capsaicinoids':      set(),
    'Betalains':          set(),
}

# Offline name-substring fallback, used ONLY for the three groups above that
# have no clean ChEBI structural class.  Substrings are matched against the
# lower-cased original compound name.
GROUP_NAME_HINTS = {
    'Curcuminoids':  ['curcumin', 'demethoxycurcumin', 'bisdemethoxycurcumin'],
    'Capsaicinoids': ['capsaicin', 'dihydrocapsaicin', 'nordihydrocapsaicin',
                      'nonivamide'],
    'Betalains':     ['betalain', 'betacyanin', 'betaxanthin', 'betanin',
                      'betanidin', 'vulgaxanthin', 'indicaxanthin'],
}


# ===========================================================================
# 4.  Helpers
# ===========================================================================
def normalize_name(s):
    """Lower-case, strip everything but a-z0-9 so 'gallic_acid', 'Gallic acid'
    and 'gallic-acid' all collapse to 'gallicacid'."""
    if not s:
        return ''
    return re.sub(r'[^a-z0-9]', '', str(s).lower())


def _normalize_cas(c):
    if not c:
        return ''
    return re.sub(r'[^0-9]', '', str(c)).lstrip('0')


def _parse_chebi(chebi):
    """Accept 30778, '30778', 'CHEBI:30778' -> 30778 (int) or None."""
    if chebi is None:
        return None
    if isinstance(chebi, int):
        return chebi
    m = re.search(r'(\d+)', str(chebi))
    return int(m.group(1)) if m else None


def build_chem_meta(feedchem_entries):
    """Build {normalized_name: {'name','cas','chebi'}} from a feedstock
    'chems' list (entries are dicts with name/cas/chebi, or bare name strings)."""
    meta = {}
    for e in feedchem_entries:
        if isinstance(e, dict):
            name = e.get('name')
            meta[normalize_name(name)] = {
                'name': name, 'cas': e.get('cas'), 'chebi': e.get('chebi')}
        else:
            meta[normalize_name(e)] = {'name': e, 'cas': None, 'chebi': None}
    return meta


# ===========================================================================
# 5.  ChEBI ancestry resolver
# ===========================================================================
# Backends, tried in this order on the first cache-miss:
#   1. A local ChEBI OBO file, read by a tiny built-in is_a parser
#        (RECOMMENDED; no third-party packages, no FTP, parses the 53 MB
#         chebi_lite.obo in a few seconds).  Download once over HTTPS:
#          https://ftp.ebi.ac.uk/pub/databases/chebi/ontology/chebi_lite.obo
#        Auto-detected if named `chebi_lite.obo`/`chebi.obo` next to these
#        scripts (or set env var CHEBI_OBO_PATH).
#   2. `libchebipy` (pip install libchebipy) — convenient, but its first
#        lookup downloads via ftp://ftp.ebi.ac.uk, which many networks block.
# Resolved ancestor sets are cached to `cache_path` (JSON); once the cache
# covers the compounds in use, neither backend is touched again.
def _build_isa_index(path):
    """Stream an OBO file and return {int_chebi_id: set(int parent ids)}
    following only `is_a` lines.  Lightweight: one pass, no object model."""
    parents = {}
    cur = None
    id_re = re.compile(r'CHEBI:(\d+)')
    with open(path, 'r', encoding='utf-8', errors='replace') as fh:
        for line in fh:
            if line.startswith('['):           # new stanza ([Term]/[Typedef])
                cur = None
            elif line.startswith('id:'):
                m = id_re.search(line)
                cur = int(m.group(1)) if m else None
                if cur is not None:
                    parents.setdefault(cur, set())
            elif cur is not None and line.startswith('is_a:'):
                m = id_re.search(line)
                if m:
                    parents[cur].add(int(m.group(1)))
    return parents


def _obo_names(path, ids):
    """One pass over the OBO returning {int_id: name} for the requested ids."""
    want = set(int(i) for i in ids)
    names = {}
    cur = None
    id_re = re.compile(r'CHEBI:(\d+)')
    with open(path, 'r', encoding='utf-8', errors='replace') as fh:
        for line in fh:
            if line.startswith('['):
                cur = None
            elif line.startswith('id:'):
                m = id_re.search(line)
                cur = int(m.group(1)) if m else None
            elif cur is not None and cur in want and line.startswith('name:'):
                names[cur] = line.split(':', 1)[1].strip()
    return names


def _default_obo_path(cache_path=None):
    candidates = []
    env = os.environ.get('CHEBI_OBO_PATH')
    if env:
        candidates.append(env)
    dirs = []
    if cache_path:
        dirs.append(os.path.dirname(os.path.abspath(cache_path)))
    dirs.append(os.getcwd())
    for d in dirs:
        candidates.append(os.path.join(d, 'chebi_lite.obo'))
        candidates.append(os.path.join(d, 'chebi.obo'))
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return None


class ChebiResolver:
    """Resolves the transitive set of `is_a` ancestor ChEBI ids for a compound.

    If no backend and no cache are available the resolver degrades gracefully:
    `available` is False and group matching simply does not fire (exact-name
    matching and the name-hint fallback still work).
    """

    def __init__(self, cache_path=None, obo_path=None, enable=True):
        self.cache_path = cache_path
        self._anc = {}            # int chebi_id -> frozenset(int ancestors)
        self.error = None
        self._dirty = False
        self._load_disk_cache()

        self._obo_path = obo_path or _default_obo_path(cache_path)
        self._isa = None          # lazily-built {id: set(parents)} from OBO
        self._lib = None          # lazily-imported libchebipy
        self._backend = None      # 'obo' | 'libchebipy' | None
        self._backend_ready = False
        self._enable = enable

        # Best-effort availability flag (without doing the heavy load yet).
        lib_ok = False
        if enable and self._obo_path is None:
            try:
                import importlib.util
                lib_ok = importlib.util.find_spec('libchebipy') is not None
            except Exception:
                lib_ok = False
        self.available = (
            (enable and self._obo_path is not None) or lib_ok or bool(self._anc))

    # -- disk cache ---------------------------------------------------------
    def _load_disk_cache(self):
        if self.cache_path and os.path.exists(self.cache_path):
            try:
                with open(self.cache_path) as fh:
                    raw = json.load(fh)
                self._anc = {int(k): frozenset(v) for k, v in raw.items()}
            except Exception:
                self._anc = {}

    def save_cache(self):
        if self.cache_path and self._dirty:
            try:
                with open(self.cache_path, 'w') as fh:
                    json.dump({str(k): sorted(v) for k, v in self._anc.items()}, fh)
                self._dirty = False
            except Exception:
                pass

    # -- backend (loaded lazily, only on a cache miss) ----------------------
    def _ensure_backend(self):
        if self._backend_ready:
            return
        self._backend_ready = True
        if not self._enable:
            return
        # 1) OBO file, parsed by a tiny built-in is_a reader (no dependencies,
        #    parses chebi_lite.obo in a few seconds with minimal memory).
        if self._obo_path and os.path.exists(self._obo_path):
            try:
                self._isa = _build_isa_index(self._obo_path)
                self._backend = 'obo'
                return
            except Exception as exc:
                self.error = f"Could not read OBO '{self._obo_path}': {exc}"
        # 2) libchebipy (downloads via ftp:// on first use)
        try:
            import libchebipy
            self._lib = libchebipy
            self._backend = 'libchebipy'
        except Exception as exc:
            self.error = (self.error or '') + f" libchebipy unavailable: {exc}"

    def _ancestors_obo(self, cid):
        """Transitive is_a ancestors from the prebuilt index (includes self)."""
        parents = self._isa
        seen = {cid}
        stack = [cid]
        while stack:
            c = stack.pop()
            for p in parents.get(c, ()):
                if p not in seen:
                    seen.add(p)
                    stack.append(p)
        return seen

    def _ancestors_lib(self, cid):
        seen, stack = set(), [cid]
        try:
            while stack:
                c = stack.pop()
                if c in seen:
                    continue
                seen.add(c)
                ent = self._lib.ChebiEntity('CHEBI:%d' % c)
                for rel in ent.get_outgoings():
                    if rel.get_type() == 'is_a':
                        tgt = _parse_chebi(rel.get_target_chebi_id())
                        if tgt is not None and tgt not in seen:
                            stack.append(tgt)
        except Exception as exc:            # network dependent
            self.error = f"ChEBI lookup failed for CHEBI:{cid}: {exc}"
            seen = {cid}
        return seen

    # -- ancestry -----------------------------------------------------------
    def ancestors(self, chebi_id):
        """Return frozenset of ancestor ids (including the id itself) following
        only `is_a` relations.  Empty frozenset if unresolved."""
        cid = _parse_chebi(chebi_id)
        if cid is None:
            return frozenset()
        cached = self._anc.get(cid)
        # Trust a non-singleton cached entry. A singleton {cid} means a prior
        # lookup failed (e.g. blocked FTP) and poisoned the cache, so we
        # re-resolve it rather than trust it.
        if cached is not None and cached != frozenset({cid}):
            return cached
        self._ensure_backend()
        if self._backend == 'obo':
            seen = self._ancestors_obo(cid)
        elif self._backend == 'libchebipy':
            seen = self._ancestors_lib(cid)
        else:
            return cached if cached is not None else frozenset()
        # keep `available` honest now that a backend has actually run
        self.available = True
        result = frozenset(seen)
        self._anc[cid] = result
        self._dirty = True
        return result


# ===========================================================================
# 6.  Classifier:  compound -> {category: match_basis}
# ===========================================================================
class ThresholdClassifier:
    def __init__(self, chebi=None, use_name_hints=True):
        self.chebi = chebi                  # a ChebiResolver, or None
        self.use_name_hints = use_name_hints
        self._by_name = {}
        self._by_cas = {}
        self._by_chebi = {}
        for cat, spec in NAMED_COMPOUNDS.items():
            for n in spec.get('names', []):
                self._by_name[normalize_name(n)] = cat
            for c in spec.get('cas', []):
                self._by_cas[_normalize_cas(c)] = cat
            for ch in spec.get('chebi', []):
                self._by_chebi[int(ch)] = cat
        self._cache = {}

    def classify(self, name=None, cas=None, chebi=None):
        """Return {category: 'exact'|'group'|'group-name'}.

        Exact matches short-circuit and take precedence over group membership.
        """
        key = (normalize_name(name), _normalize_cas(cas), _parse_chebi(chebi))
        if key in self._cache:
            return self._cache[key]

        result = {}

        # 1. exact named-compound match (name / CAS / ChEBI)
        exact = (self._by_name.get(key[0])
                 or self._by_cas.get(key[1])
                 or (self._by_chebi.get(key[2]) if key[2] is not None else None))
        if exact:
            result = {exact: 'exact'}
            self._cache[key] = result
            return result

        # 2. group membership through the ChEBI is_a hierarchy
        if key[2] is not None and self.chebi is not None:
            anc = self.chebi.ancestors(key[2])
            if anc:
                for cat, ids in GROUP_CLASSES.items():
                    if ids and (ids & anc):
                        result[cat] = 'group'

        # 3. offline name-substring fallback for groups ChEBI can't model
        if self.use_name_hints and name:
            nlow = str(name).lower()
            for cat, hints in GROUP_NAME_HINTS.items():
                if cat not in result and any(h in nlow for h in hints):
                    result[cat] = 'group-name'

        self._cache[key] = result
        return result


# ===========================================================================
# 7.  Threshold resolution + evaluation
# ===========================================================================
def governing_threshold(categories, column):
    """Most conservative (lowest) numeric threshold among `categories` for
    `column`.  Returns (value, [categories that set it]); (None, []) if none
    of the matched categories has a numeric threshold for that column."""
    vals = [(THRESHOLDS[c][column], c)
            for c in categories
            if c in THRESHOLDS and THRESHOLDS[c].get(column) is not None]
    if not vals:
        return None, []
    lo = min(v for v, _ in vals)
    return lo, [c for v, c in vals if v == lo]


@dataclass
class Flag:
    chemical: str
    unit: str
    column: str               # 'water' | 'other_solvent' | 'drying'
    column_label: str
    exposure_C: float
    threshold_C: float
    exceedance_C: float
    governed_by: str          # category whose threshold was applied
    matched_categories: str   # all categories the compound matched
    match_basis: str          # exact | group | group-name


def evaluate_temperature_flags(exposures, classifier, tol=1e-6):
    """Evaluate a set of unit exposures and return (flags, unclassified_names).

    `exposures` is a list of dicts:
        {
          'unit':   'Extractor (E201)',
          'column': 'water' | 'other_solvent' | 'drying',
          'temp_C': 98.0,
          'chemicals': [ {'name':..,'cas':..,'chebi':..}, ... ],
        }

    A flag is raised when the unit temperature is strictly above the governing
    threshold for a matched compound.
    """
    flags = []
    unclassified = set()
    for ex in exposures:
        col = ex['column']
        temp = ex['temp_C']
        unit = ex['unit']
        if temp is None:
            continue
        for chem in ex.get('chemicals', []):
            cats = classifier.classify(
                name=chem.get('name'), cas=chem.get('cas'),
                chebi=chem.get('chebi'))
            if not cats:
                if chem.get('name'):
                    unclassified.add(chem['name'])
                continue
            thr, gov = governing_threshold(cats.keys(), col)
            if thr is None:
                continue
            if temp > thr + tol:
                basis = cats.get(gov[0], 'group') if gov else 'group'
                flags.append(Flag(
                    chemical=chem.get('name'),
                    unit=unit,
                    column=col,
                    column_label=COLUMN_LABEL.get(col, col),
                    exposure_C=round(float(temp), 1),
                    threshold_C=float(thr),
                    exceedance_C=round(float(temp) - float(thr), 1),
                    governed_by=", ".join(gov),
                    matched_categories=", ".join(sorted(cats.keys())),
                    match_basis=basis,
                ))
    return flags, sorted(unclassified)


def flags_to_rows(flags):
    """List[Flag] -> list of plain dicts (handy for a pandas DataFrame)."""
    return [asdict(f) for f in flags]


def _operation_phrase(unit):
    """Map a unit label to the process operation it represents."""
    u = str(unit).lower()
    if 'extract' in u:
        return 'extraction'
    if 'evaporat' in u:
        return 'solvent evaporation'
    if 'dry' in u:
        return 'spray drying'
    return str(unit)


def _join_clauses(items):
    """['a'] -> 'a';  ['a','b'] -> 'a and b';  ['a','b','c'] -> 'a, b and c'."""
    items = [i for i in items if i]
    if not items:
        return ''
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])} and {items[-1]}"


def _fmt_temp(v):
    """120.0 -> '120';  78.4 -> '78.4'."""
    v = float(v)
    return f"{v:.0f}" if abs(v - round(v)) < 0.05 else f"{v:.1f}"


def narrative_lines(flags):
    """Aggregate flags per compound into plain-language sentences, e.g.:

        "Quercetin exceeded a temperature known to cause degradation during
         solvent evaporation (100 °C vs 90 °C limit) and spray drying
         (110 °C vs 60 °C limit). The real extract would be expected to contain
         lower quantities of this compound than the simulation predicts.
         (Flagged by group: Flavonoids and Phenolic Compounds.)"
    """
    by_comp = {}
    for f in flags:
        by_comp.setdefault(f.chemical, []).append(f)

    lines = []
    for comp in sorted(by_comp,
                       key=lambda c: -max(x.exceedance_C for x in by_comp[c])):
        fl = sorted(by_comp[comp], key=lambda x: -x.exceedance_C)
        clauses = [f"{_operation_phrase(f.unit)} "
                   f"({_fmt_temp(f.exposure_C)} °C vs {_fmt_temp(f.threshold_C)} "
                   f"°C limit)" for f in fl]
        sentence = (
            f"{comp} exceeded a temperature known to cause degradation during "
            f"{_join_clauses(clauses)}. The real extract would be expected to "
            f"contain lower quantities of this compound than the simulation "
            f"predicts.")
        # Note the basis only when matched by group (for an exact-named
        # compound the name already says why it was flagged).
        group_cats = []
        for f in fl:
            if f.match_basis != 'exact':
                for c in f.governed_by.split(', '):
                    if c and c not in group_cats:
                        group_cats.append(c)
        if group_cats:
            sentence += f" (Flagged by group: {_join_clauses(group_cats)}.)"
        lines.append(sentence)
    return lines


def format_flags_text(flags, unclassified=None, n_unclassified_note=True):
    """Render flags as a plain-language text block for stdout."""
    lines = []
    if not flags:
        lines.append("  No compounds were held above their temperature "
                     "thresholds.")
    else:
        n = len({f.chemical for f in flags})
        lines.append(f"  {n} compound(s) exceeded a degradation threshold:\n")
        for sentence in narrative_lines(flags):
            # wrap each sentence as an indented bullet
            lines.append("  • " + sentence)
            lines.append("")
        if lines and lines[-1] == "":
            lines.pop()
    if unclassified and n_unclassified_note:
        lines.append("")
        lines.append(f"  ({len(unclassified)} handled compound(s) had no "
                     f"threshold-table match and were not checked.)")
    return "\n".join(lines)


# ===========================================================================
# 9.  Command-line self-test / diagnostic
# ===========================================================================
# Usage:
#   python temperature_thresholds.py                 # uses libchebipy / auto OBO
#   python temperature_thresholds.py --obo chebi_lite.obo
#   python temperature_thresholds.py --obo chebi_lite.obo --chebi CHEBI:16243
#
# A quick way to confirm group matching works in your environment before
# running the full TEA, and to pre-build chebi_group_cache.json.
if __name__ == '__main__':
    import sys
    import time

    argv = sys.argv[1:]
    obo = None
    if '--obo' in argv:
        i = argv.index('--obo')
        obo = argv[i + 1]
        del argv[i:i + 2]
    ids = []
    if '--chebi' in argv:
        i = argv.index('--chebi')
        ids = [a for a in argv[i + 1:] if not a.startswith('--')]
    trace = None
    if '--trace' in argv:
        i = argv.index('--trace')
        trace = argv[i + 1]

    resolver = ChebiResolver(cache_path='chebi_group_cache.json', obo_path=obo)
    print(f"Backend OBO path  : {resolver._obo_path or '(none - will try libchebipy)'}",
          flush=True)
    print(f"Reported available: {resolver.available}", flush=True)

    # --trace shows the full is_a ancestry of one compound (for debugging
    # "why didn't X match group Y?").
    if trace:
        anc = sorted(resolver.ancestors(trace))
        names = (_obo_names(resolver._obo_path, anc)
                 if resolver._obo_path else {})
        print(f"\nis_a ancestry of {trace}  ({len(anc)} classes):", flush=True)
        for a in anc:
            tag = ''
            for grp, gids in GROUP_CLASSES.items():
                if a in gids:
                    tag = f"   <-- defines group '{grp}'"
            print(f"  CHEBI:{a:<7d} {names.get(a, ''):<34s}{tag}", flush=True)
        resolver.save_cache()
        sys.exit(0)

    clf = ThresholdClassifier(chebi=resolver)

    if ids:
        probes = [(i, None, i) for i in ids]
    else:
        # representative compounds spanning several groups
        probes = [
            ('quercetin', '117-39-5', 'CHEBI:16243'),
            ('gallic acid', '149-91-7', 'CHEBI:30778'),
            ('catechin', '154-23-4', 'CHEBI:15600'),
            ('resveratrol', '501-36-0', 'CHEBI:27881'),
            ('linoleic acid', '60-33-3', 'CHEBI:17351'),
            ('lutein', '127-40-2', 'CHEBI:28838'),
            ('lycopene', '502-65-8', 'CHEBI:15948'),
        ]

    if resolver._obo_path:
        print(f"\nReading {os.path.basename(resolver._obo_path)} (one-time, "
              f"a few seconds)...", end='', flush=True)
    t0 = time.time()
    first = clf.classify(name=probes[0][0], cas=probes[0][1], chebi=probes[0][2])
    if resolver._obo_path:
        print(f" done in {time.time() - t0:.1f}s.", flush=True)

    print("\n%-16s %-14s %s" % ('name', 'chebi', 'matched categories'), flush=True)
    print('-' * 70, flush=True)
    print("%-16s %-14s %s" % (probes[0][0], probes[0][2], first or '(none)'),
          flush=True)
    for name, cas, chebi in probes[1:]:
        cats = clf.classify(name=name, cas=cas, chebi=chebi)
        print("%-16s %-14s %s" % (name, chebi, cats or '(none)'), flush=True)
    resolver.save_cache()
    print("\nWrote cache: chebi_group_cache.json", flush=True)
    if resolver.error:
        print("note:", resolver.error, flush=True)