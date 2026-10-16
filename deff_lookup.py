"""
Deff/Dab (porosity/tortuosity factor X) lookup for the extraction kinetics.

The Reddy-Doraiswamy rate constant is k1 = R^2 / (15 * Dab * X * Cs), where
X = Deff/Dab is a property of the solid matrix. X is read from the "Overal"
sheet of Food_byproduct_porosity.xlsx:

    Material | Category | ε | ε (as reported) | Deff/Dab (ε^1.5) | Source

Lookup order for a feedstock:
  1. exact   - the feedstock name matches a Material row -> that row's Deff/Dab
  2. category - average for the feedstock's category, computed from the rows
                in that category (same rule as the sheet's "Category average"
                tab: average ε, then ε_avg^1.5; if any row in the category is
                not on the ε^1.5 basis -- e.g. woody biomass, which uses ε/τ --
                the Deff/Dab column itself is averaged instead)
  3. stand-in - the category has no rows but CATEGORY_STAND_INS names another
               category to use for now (hulls -> fresh fruits & vegetables)
  4. overall - the category has no rows in the sheet and no stand-in: average of
               every material in the sheet, computed the same way. This is a
               placeholder and is flagged so it can be replaced with data.

New rows added to the "Overal" sheet are picked up automatically.
"""

import os
import re

import pandas as pd

DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'Food_byproduct_porosity.xlsx')
SHEET = 'Overal'

# Feedstock 'type' strings used in the tool -> category names in the sheet
# (both sides normalised by _norm before comparison).
CATEGORY_ALIASES = {
    'fresh fruit and veg': 'fresh fruits and vegetables',
    'fresh fruits and veg': 'fresh fruits and vegetables',
    'fruit pomaces': 'pomaces',
    'pomace': 'pomaces',
    'shell': 'shells',
    'hull': 'hulls',
    'leaf': 'leaves',
    'wood': 'woody biomass',
}


# Stand-in categories: if a category has no rows in the sheet, use the
# average of this other category instead. Applies only while the category is
# empty -- as soon as rows are added for it, its own average is used.
CATEGORY_STAND_INS = {
    'hulls': 'fresh fruits and vegetables',   # assumption until hull data exists
}


def _norm(s):
    s = str(s).strip().lower().replace('&', ' and ')
    s = re.sub(r'[^a-z0-9]+', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()


def _num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if v == v else None   # drop NaN


def load_deff_table(path=DEFAULT_PATH):
    """Read the sheet into a DataFrame with columns
    material, category, eps, deff, source (+ normalised keys)."""
    raw = pd.read_excel(path, sheet_name=SHEET)
    raw.columns = [str(c).strip() for c in raw.columns]

    def col(prefix):
        for c in raw.columns:
            if c.lower().startswith(prefix):
                return c
        raise KeyError(f"No column starting with '{prefix}' in sheet "
                       f"'{SHEET}' of {path}")

    c_mat, c_cat = col('material'), col('category')
    c_eps = next(c for c in raw.columns if c in ('ε', 'ε (midpoint)')
                 or (c.startswith('ε') and 'reported' not in c))
    c_deff, c_src = col('deff'), col('source')

    rows = []
    for _, r in raw.iterrows():
        mat, cat = r[c_mat], r[c_cat]
        deff = _num(r[c_deff])
        # The notes block below the table has no category / no number.
        if pd.isna(mat) or pd.isna(cat) or deff is None:
            continue
        rows.append({'material': str(mat).strip(),
                     'category': str(cat).strip(),
                     'eps': _num(r[c_eps]),
                     'deff': deff,
                     'source': '' if pd.isna(r[c_src]) else str(r[c_src]),
                     'material_key': _norm(mat),
                     'category_key': _norm(cat)})
    if not rows:
        raise ValueError(f"No usable rows in sheet '{SHEET}' of {path}")
    return pd.DataFrame(rows)


def _group_average(df):
    """Average Deff/Dab for a set of rows (see module docstring)."""
    eps = df['eps']
    on_eps15 = (eps.notna().all() and
                ((eps ** 1.5 - df['deff']).abs() <= 1e-6 * df['deff']).all())
    if on_eps15:
        eps_avg = float(eps.mean())
        return eps_avg ** 1.5, eps_avg, 'average ε, then ε^1.5'
    return float(df['deff'].mean()), (float(eps.mean()) if eps.notna().all()
                                      else None), 'average of Deff/Dab'


def lookup_deff(table, material, category):
    """Return a dict: value, basis ('exact'|'category'|'overall'),
    matched, n_rows, eps, method, sources, note."""
    m_key = _norm(material)
    c_key = _norm(category)
    c_key = CATEGORY_ALIASES.get(c_key, c_key)

    hit = table[table['material_key'] == m_key]
    if len(hit):
        r = hit.iloc[0]
        return {'value': float(r['deff']), 'basis': 'exact',
                'matched': r['material'], 'n_rows': 1, 'eps': r['eps'],
                'method': 'sheet value', 'sources': [r['source']],
                'note': f"exact match: {r['material']}"}

    grp = table[table['category_key'] == c_key]
    if len(grp):
        val, eps, how = _group_average(grp)
        return {'value': val, 'basis': 'category',
                'matched': grp['category'].iloc[0], 'n_rows': len(grp),
                'eps': eps, 'method': how,
                'sources': sorted(set(grp['source'])),
                'note': (f"category average of {len(grp)} material(s) in "
                         f"'{grp['category'].iloc[0]}' ({how})")}

    stand_in = CATEGORY_STAND_INS.get(c_key)
    grp = table[table['category_key'] == stand_in] if stand_in else grp
    if len(grp):
        val, eps, how = _group_average(grp)
        sname = grp['category'].iloc[0]
        return {'value': val, 'basis': 'stand-in category',
                'matched': sname, 'n_rows': len(grp),
                'eps': eps, 'method': how,
                'sources': sorted(set(grp['source'])),
                'note': (f"no data for '{category}' yet; assumed same as "
                         f"'{sname}' (average of {len(grp)} material(s), "
                         f"{how})")}

    val, eps, how = _group_average(table)
    return {'value': val, 'basis': 'overall', 'matched': None,
            'n_rows': len(table), 'eps': eps, 'method': how,
            'sources': sorted(set(table['source'])),
            'note': (f"no data for '{material}' or category '{category}'; "
                     f"using the average of all {len(table)} materials "
                     f"({how}) as a placeholder")}
