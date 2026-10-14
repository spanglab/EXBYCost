"""
Protein / lipid accounting for feedstock compositions.

A feedstock entry lists individually measured compounds (from the byproduct
database) plus, optionally, a reported TOTAL protein and TOTAL lipid on the
same dry basis. This module:

  1. Classifies every listed compound as 'protein', 'lipid' or 'other',
     primarily from its ChEBI ontology ancestry:
        protein <- amino acid (CHEBI:33709), peptide (CHEBI:16670),
                   protein (CHEBI:36080)
        lipid   <- fatty acid (CHEBI:35366), glyceride (CHEBI:47778),
                   and, with lipid_scope='all', anything under lipid
                   (CHEBI:18059): sterols, carotenoids, triterpenes, ...
     with a name-keyword fallback for compounds that have no ChEBI ID or
     whose ChEBI ancestry gives no class (e.g. tocopherols, which ChEBI
     files under 'vitamin E' rather than 'lipid').
  2. Sums the measured protein and lipid compounds.
  3. Compares each sum with the reported total:
        measured >= total -> keep the measured compounds as they are,
                             add nothing (the report flags the excess);
        measured <  total -> add the shortfall as a model compound
                             (glutathione for protein, trilinolein for
                             lipid).
  4. Recomputes the balance compound (Cellulose by default) last, as
     1 - (everything else), so the composition always sums to 1. The
     balance only closes the mass balance; its listed value is ignored.

Feedstock entry keys used
-------------------------
    'chems'          list of {'name', 'cas', 'chebi'} dicts (as before)
    'comp'           dry-basis mass fractions, same order (as before)
    'total_protein'  optional, dry-basis mass fraction (e.g. 0.205)
    'total_lipid'    optional, dry-basis mass fraction (e.g. 0.110)
    'balance'        optional, name of the balance compound
                     (default 'Cellulose')
A chem dict may also carry 'class': 'protein' | 'lipid' | 'other' to
override the automatic classification for that compound.

ChEBI ancestry sources, tried in order
--------------------------------------
    a. the JSON cache (chebi_macro_cache.json next to this file);
    b. a local ChEBI .obo file: $CHEBI_OBO_PATH, or chebi_lite.obo /
       chebi.obo / chebi_core.obo / chebi_slim.obo next to this file
       (the same file temperature_thresholds can use);
    c. the EBI OLS4 web API.
Only successful lookups are cached, so after the first run no file
parsing or network access is needed.
"""

import json
import os
import urllib.parse
import urllib.request

_HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.join(_HERE, 'chebi_macro_cache.json')
_CACHE_VERSION = 1

# ChEBI classes that decide the category
AMINO_ACID = 'CHEBI:33709'
PEPTIDE = 'CHEBI:16670'
PROTEIN = 'CHEBI:36080'
FATTY_ACID = 'CHEBI:35366'
GLYCERIDE = 'CHEBI:47778'
LIPID = 'CHEBI:18059'
_ROOTS = (AMINO_ACID, PEPTIDE, PROTEIN, FATTY_ACID, GLYCERIDE, LIPID)

# Model compounds used to fill a shortfall (both verified to load natively
# in thermosteam, pass the proxy finder, and route through the extractor's
# HSP path -- see the tomato pomace note in extraction_tea_tool.py).
MODEL_COMPOUNDS = {
    'protein': {'name': 'glutathione', 'cas': '70-18-8',
                'chebi': 'CHEBI:16856'},
    'lipid':   {'name': 'trilinolein', 'cas': '537-40-6', 'chebi': None},
}

# ---------------------------------------------------------------------------
# Name-keyword fallback
# ---------------------------------------------------------------------------
_AMINO_ACID_NAMES = {
    'alanine', 'arginine', 'asparagine', 'aspartic acid', 'aspartate',
    'cysteine', 'cystine', 'glutamic acid', 'glutamate', 'glutamine',
    'glycine', 'histidine', 'isoleucine', 'leucine', 'lysine', 'methionine',
    'phenylalanine', 'proline', 'hydroxyproline', 'serine', 'threonine',
    'tryptophan', 'tyrosine', 'valine', 'ornithine', 'citrulline',
}
_PROTEIN_KEYWORDS = ('peptide', 'protein', 'albumin', 'globulin', 'glutelin',
                     'prolamin', 'glutathione', 'zein', 'gluten', 'casein')
_LIPID_KEYWORDS = (
    # fatty acids
    'fatty acid', 'palmit', 'stearic', 'stearate', 'oleic', 'oleate',
    'linoleic', 'linolenic', 'myristic', 'lauric', 'capric', 'caprylic',
    'arachid', 'behenic', 'erucic', 'hexadecanoic', 'octadecanoic',
    'eicosa', 'docosa', 'tetradecanoic', 'dodecanoic',
    # glycerides / complex lipids
    'glyceride', 'olein', 'stearin', 'palmitin', 'linolein', 'phospholipid',
    'lecithin', 'phosphatidyl', 'sphingo', 'ceramide',
    # isoprenoid / sterol lipids
    'sterol', 'stanol', 'tocopherol', 'tocotrienol', 'carotene', 'lycopene',
    'neurosporene', 'squalene', 'oryzanol', 'phytol', 'lutein', 'amyrin',
    'cholest', 'lanost', 'campest', 'sitost', 'stigmast', 'cycloarten',
)


def _strip_stereo(name):
    n = name.lower().replace('_', ' ').replace('-', ' ').strip()
    for prefix in ('l ', 'd ', 'dl '):
        if n.startswith(prefix):
            n = n[len(prefix):]
    return n


def classify_by_name(name):
    """'protein' | 'lipid' | None, from the compound name alone."""
    n = _strip_stereo(name)
    if n in _AMINO_ACID_NAMES:
        return 'protein'
    low = name.lower()
    if any(k in low for k in _PROTEIN_KEYWORDS):
        return 'protein'
    if any(k in low for k in _LIPID_KEYWORDS):
        return 'lipid'
    if 'xanthin' in low and not low.endswith('xanthine'):  # carotenoids,
        return 'lipid'                                       # not purines
    return None


# ---------------------------------------------------------------------------
# ChEBI ancestry
# ---------------------------------------------------------------------------
def _norm_chebi(chebi):
    if not chebi:
        return None
    c = str(chebi).strip().upper()
    if not c.startswith('CHEBI:'):
        c = 'CHEBI:' + c
    return c


class ChebiAncestry:
    """Answers 'which of _ROOTS is this ChEBI ID under?' with caching."""

    def __init__(self, cache_path=CACHE_PATH, use_network=True,
                 obo_path=None, timeout=15):
        self.cache_path = cache_path
        self.use_network = use_network
        self.timeout = timeout
        self._obo_path = obo_path
        self._obo_parents = None      # parsed lazily, only if needed
        self._dirty = False
        self.cache = {}
        try:
            with open(cache_path) as fh:
                data = json.load(fh)
            if data.get('version') == _CACHE_VERSION:
                self.cache = data.get('roots', {})
        except Exception:
            pass

    # -- local .obo --------------------------------------------------------
    def _find_obo(self):
        if self._obo_path:
            return self._obo_path
        env = os.environ.get('CHEBI_OBO_PATH')
        if env and os.path.exists(env):
            return env
        for fn in ('chebi_lite.obo', 'chebi.obo', 'chebi_core.obo',
                   'chebi_slim.obo'):
            p = os.path.join(_HERE, fn)
            if os.path.exists(p):
                return p
        return None

    def _load_obo(self):
        if self._obo_parents is not None:
            return self._obo_parents
        self._obo_parents = {}
        path = self._find_obo()
        if not path:
            return self._obo_parents
        cur = None
        try:
            with open(path, encoding='utf-8') as fh:
                for line in fh:
                    if line.startswith('[Term]'):
                        cur = None
                    elif line.startswith('id: '):
                        cur = line[4:].strip()
                        self._obo_parents.setdefault(cur, set())
                    elif cur and line.startswith('is_a: '):
                        self._obo_parents[cur].add(line[6:].split()[0])
        except Exception:
            self._obo_parents = {}
        return self._obo_parents

    def _roots_from_obo(self, cid):
        parents = self._load_obo()
        if cid not in parents:
            return None
        seen, stack = set(), [cid]
        while stack:
            for p in parents.get(stack.pop(), ()):
                if p not in seen:
                    seen.add(p)
                    stack.append(p)
        seen.add(cid)
        return [r for r in _ROOTS if r in seen]

    # -- OLS4 web API ------------------------------------------------------
    def _roots_from_ols(self, cid):
        if not self.use_network:
            return None
        iri = 'http://purl.obolibrary.org/obo/' + cid.replace(':', '_')
        enc = urllib.parse.quote(urllib.parse.quote(iri, safe=''), safe='')
        base = (f'https://www.ebi.ac.uk/ols4/api/ontologies/chebi/terms/'
                f'{enc}/ancestors?size=500')
        found, page, n_pages = {cid}, 0, 1
        try:
            while page < n_pages:
                req = urllib.request.Request(
                    f'{base}&page={page}',
                    headers={'Accept': 'application/json',
                             'User-Agent': 'EXBYCost/macronutrients'})
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    data = json.loads(r.read().decode())
                for t in data.get('_embedded', {}).get('terms', []):
                    if t.get('obo_id'):
                        found.add(t['obo_id'])
                n_pages = data.get('page', {}).get('totalPages', 1) or 1
                page += 1
        except Exception:
            return None
        return [r for r in _ROOTS if r in found]

    def roots(self, chebi):
        """List of decisive ChEBI classes above `chebi`, or None if unknown."""
        cid = _norm_chebi(chebi)
        if cid is None:
            return None
        if cid in self.cache:
            return self.cache[cid]
        res = self._roots_from_obo(cid)
        if res is None:
            res = self._roots_from_ols(cid)
        if res is not None:
            self.cache[cid] = res
            self._dirty = True
        return res

    def save(self):
        if not self._dirty:
            return
        try:
            with open(self.cache_path, 'w') as fh:
                json.dump({'version': _CACHE_VERSION, 'roots': self.cache},
                          fh, indent=0, sort_keys=True)
            self._dirty = False
        except Exception:
            pass


def _class_from_roots(roots, lipid_scope):
    if roots is None:
        return None
    r = set(roots)
    if FATTY_ACID in r or GLYCERIDE in r:
        return 'lipid'
    if r & {AMINO_ACID, PEPTIDE, PROTEIN}:
        return 'protein'
    if LIPID in r and lipid_scope == 'all':
        return 'lipid'
    return None


def classify_compound(entry, resolver, lipid_scope='all'):
    """Return (class, source) for one chem dict/str.

    class is 'protein', 'lipid' or 'other'; source says how it was decided.
    """
    if isinstance(entry, str):
        entry = {'name': entry}
    name = entry.get('name', '')
    override = entry.get('class')
    if override in ('protein', 'lipid', 'other'):
        return override, 'override'

    chebi = entry.get('chebi')
    roots = resolver.roots(chebi) if chebi else None
    cls = _class_from_roots(roots, lipid_scope)
    if cls is not None:
        return cls, 'ChEBI'

    by_name = classify_by_name(name)
    if lipid_scope == 'fatty' and by_name == 'lipid':
        # Only fatty acids / glycerides count in the narrow scope.
        low = name.lower()
        fatty = ('sterol', 'stanol', 'tocopherol', 'tocotrienol', 'carotene',
                 'lycopene', 'neurosporene', 'squalene', 'oryzanol', 'phytol',
                 'lutein', 'amyrin', 'xanthin', 'cholest', 'lanost', 'campest',
                 'sitost', 'stigmast', 'cycloarten')
        if any(k in low for k in fatty):
            by_name = None
    if by_name is not None:
        return by_name, 'name'
    if chebi and roots is None:
        return 'other', 'ChEBI unavailable'
    return 'other', ('ChEBI' if chebi else 'name')


def classify_feedstock(fs, resolver=None, lipid_scope='all'):
    """{compound name: 'protein' | 'lipid' | 'other'} for every compound in
    a feedstock entry, plus the model compounds (glutathione -> protein,
    trilinolein -> lipid) so filled-in amounts are counted too.

    Works whether or not the feedstock has reported totals; used to sum the
    protein and lipid carried by any process stream (e.g. the residual).
    """
    own_resolver = resolver is None
    if own_resolver:
        resolver = ChebiAncestry()
    classes = {}
    for kind, model in MODEL_COMPOUNDS.items():
        classes[model['name']] = kind
    for c in fs['chems']:
        entry = c if isinstance(c, dict) else {'name': c}
        classes[entry['name']] = classify_compound(entry, resolver,
                                                   lipid_scope)[0]
    if own_resolver:
        resolver.save()
    return classes


# ---------------------------------------------------------------------------
# Fill-in
# ---------------------------------------------------------------------------
def complete_feedstock(fs, resolver=None, lipid_scope='all', tol=1e-6):
    """Return (new_feedstock_dict, report).

    The input dict is not modified. If the feedstock has neither
    'total_protein' nor 'total_lipid', it is returned unchanged.

    Raises ValueError if the fillers would drive the balance compound
    negative (the measured compounds + totals then exceed 100 %).
    """
    totals = {'protein': fs.get('total_protein'),
              'lipid': fs.get('total_lipid')}
    if totals['protein'] is None and totals['lipid'] is None:
        return fs, None

    own_resolver = resolver is None
    if own_resolver:
        resolver = ChebiAncestry()

    chems = [dict(c) if isinstance(c, dict) else {'name': c}
             for c in fs['chems']]
    comp = [float(x) for x in fs['comp']]
    if len(chems) != len(comp):
        raise ValueError("'chems' and 'comp' have different lengths")

    rows, measured, members = [], {'protein': 0.0, 'lipid': 0.0}, \
        {'protein': [], 'lipid': []}
    for c, x in zip(chems, comp):
        cls, src = classify_compound(c, resolver, lipid_scope)
        rows.append({'compound': c['name'], 'mass_frac': x,
                     'class': cls, 'source': src})
        if cls in measured:
            measured[cls] += x
            members[cls].append(c['name'])

    if own_resolver:
        resolver.save()

    balance = fs.get('balance', 'Cellulose')
    names = [c['name'] for c in chems]
    if balance not in names:
        raise ValueError(f"balance compound '{balance}' is not in 'chems'")
    i_bal = names.index(balance)

    summary, fillers = [], {}
    for kind in ('protein', 'lipid'):
        total = totals[kind]
        if total is None:
            continue
        gap = float(total) - measured[kind]
        model = MODEL_COMPOUNDS[kind]
        if gap > tol:
            fillers[kind] = gap
            if model['name'] in names:          # merge, keep IDs unique
                j = names.index(model['name'])
                comp[j] += gap
            else:
                chems.append(dict(model))
                comp.append(gap)
                names.append(model['name'])
            action = f"added {gap * 1000:.2f} mg/g as {model['name']}"
        else:
            action = ("measured sum exceeds reported total by "
                      f"{-gap * 1000:.2f} mg/g; nothing added")
        summary.append({
            'component': kind,
            'reported_total_mg_g': float(total) * 1000,
            'measured_sum_mg_g': measured[kind] * 1000,
            'n_measured_compounds': len(members[kind]),
            'filler_mg_g': max(gap, 0.0) * 1000,
            # what the model actually carries: the measured compounds, plus
            # the filler if one was added (= max(measured, reported))
            'model_total_mg_g': (measured[kind] + max(gap, 0.0)) * 1000,
            'model_compound': model['name'] if gap > tol else '',
            'action': action,
        })

    # The balance compound only closes the mass balance: it is recomputed
    # LAST, after the protein/lipid fill-in, as 1 - (everything else). Its
    # listed value is ignored, so a listed composition that does not sum to
    # 1 is closed here too.
    listed_balance = comp[i_bal]
    comp[i_bal] = 1.0 - sum(x for j, x in enumerate(comp) if j != i_bal)
    if comp[i_bal] < -tol:
        raise ValueError(
            f"Filling protein/lipid would make '{balance}' negative "
            f"({comp[i_bal] * 1000:.1f} mg/g): the measured compounds plus "
            f"the reported totals exceed 100 % of the dry basis. Check the "
            f"totals and the measured values are on the same basis.")
    comp[i_bal] = max(comp[i_bal], 0.0)

    new_fs = dict(fs)
    new_fs['chems'] = chems
    new_fs['comp'] = comp
    report = {'summary': summary, 'compounds': rows, 'fillers': fillers,
              'balance': balance, 'balance_mg_g': comp[i_bal] * 1000,
              'listed_balance_mg_g': listed_balance * 1000,
              'lipid_scope': lipid_scope}
    return new_fs, report