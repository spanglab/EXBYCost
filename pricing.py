"""Pricing helpers for the extraction TEA tool.

Three external data sources, one module:

1. Solvent prices  -- cited base prices from solvent_prices.csv,
   optionally escalated to the present using BLS price indices via
   the FRED API (e.g. WPU0614 = PPI Basic Organic Chemicals;
   CUUR0000SEHG = CPI Water and Sewer Services).

2. California industrial retail electricity  -- pulled live from the
   U.S. EIA Open Data API v2 (state=CA, sector=IND, monthly).

3. California operator wage  -- BLS OEWS hourly mean wage for SOC
   51-8091 (Chemical Plant and System Operators), California state.
   Optionally escalated from its May reference quarter to the latest
   quarterly ECI observation via FRED ECIMANWAG (manufacturing wages),
   the methodology California EDD itself uses.

Free API keys (set as environment variables):
  FRED_API_KEY  https://fred.stlouisfed.org/docs/api/api_key.html
  EIA_API_KEY   https://www.eia.gov/opendata/register.php
  BLS_API_KEY   https://data.bls.gov/registrationEngine/

Stdlib only.  Hard errors (missing required key, API rejection, no
data) raise RuntimeError -- no silent fallback to fake numbers.  The
two *optional* escalations (FRED in get_solvent_prices, ECI in
get_california_operator_wage) degrade gracefully to the base value if
they fail, recording the reason in `meta['note']`.
"""

import csv
import json
import os
import urllib.error
import urllib.parse
import urllib.request


# ── Endpoints and constants ──────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SOLVENT_CSV = os.path.join(_HERE, 'solvent_prices.csv')

_FRED_URL = "https://api.stlouisfed.org/fred/series/observations"
_EIA_RETAIL_URL = "https://api.eia.gov/v2/electricity/retail-sales/data/"
_BLS_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"

# BLS OEWS series_id layout (25 chars):
#   OE + U + areatype(1) + area(7) + industry(6) + occupation(6) + datatype(2)
# California state, cross-industry, SOC 51-8091, hourly mean wage:
#   OE + U + S + 0600000 + 000000 + 518091 + 03
_CA_OPERATOR_HOURLY_SERIES = "OEUS060000000000051809103"
_SOC_CODE = "51-8091"
_OCCUPATION_NAME = "Chemical Plant and System Operators"

# FRED ECI: Wages & Salaries, Private Industry Workers, Manufacturing
# (quarterly SA, base Dec 2005 = 100). Used to escalate OEWS from its
# May reference to the latest quarter -- same approach as California EDD.
_ECI_SERIES = "ECIMANWAG"


# ── FRED helpers (shared by solvent and operator escalation) ─────────────
def _fred_points(series_id, api_key, start=None, end=None,
                 sort_order=None, limit=None):
    """Return ``[(date_str, value), ...]`` for a FRED series.

    Observations FRED reports as missing (the literal '.') are dropped.
    """
    params = {'series_id': series_id, 'api_key': api_key,
              'file_type': 'json'}
    if start:
        params['observation_start'] = start
    if end:
        params['observation_end'] = end
    if sort_order:
        params['sort_order'] = sort_order
    if limit:
        params['limit'] = limit
    url = _FRED_URL + '?' + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=15) as resp:
        data = json.load(resp)
    points = []
    for obs in data.get('observations', []):
        try:
            points.append((obs['date'], float(obs['value'])))
        except (ValueError, KeyError, TypeError):
            continue
    return points


def _fred_latest(series_id, api_key):
    """``(date, value)`` of the most recent valid observation."""
    pts = _fred_points(series_id, api_key, sort_order='desc', limit=3)
    return pts[0] if pts else (None, None)


def _fred_month(series_id, year, month, api_key):
    """``(date, value)`` for one specific month, or (None, None)."""
    start = f'{int(year)}-{int(month):02d}-01'
    end = f'{int(year)}-{int(month):02d}-28'
    pts = _fred_points(series_id, api_key, start=start, end=end)
    return pts[0] if pts else (None, None)


def _fred_quarter(series_id, year, quarter, api_key):
    """``(date, value)`` for a specific quarter (1-4), or (None, None).

    Quarterly FRED series are dated by the first month of the quarter
    (Q1 = YYYY-01-01, Q2 = YYYY-04-01, Q3 = YYYY-07-01, Q4 = YYYY-10-01).
    """
    start_month = {1: '01', 2: '04', 3: '07', 4: '10'}[quarter]
    start = f"{int(year)}-{start_month}-01"
    end = f"{int(year)}-{start_month}-28"
    pts = _fred_points(series_id, api_key, start=start, end=end)
    return pts[0] if pts else (None, None)


def _fred_year_average(series_id, year, api_key):
    """Calendar-year average of a FRED series, or None (fallback only)."""
    pts = _fred_points(series_id, api_key,
                       start=f'{int(year)}-01-01',
                       end=f'{int(year)}-12-31')
    vals = [v for _, v in pts]
    return sum(vals) / len(vals) if vals else None


# ── Solvent prices (CSV + optional FRED escalation) ──────────────────────
def load_price_table(csv_path=DEFAULT_SOLVENT_CSV):
    """Read solvent_prices.csv and return a list of row dicts."""
    with open(csv_path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def get_solvent_prices(csv_path=DEFAULT_SOLVENT_CSV, fred_api_key=None,
                       escalate=True):
    """Return ``(prices, meta)`` for the solvents in the CSV.

    prices : dict ``{solvent_name: price_usd_per_kg}``.
    meta   : dict ``{solvent_name: {...CSV row..., base_price,
             effective_price, escalated, note}}`` for provenance.

    Month-to-month escalation when ``escalate`` and ``fred_api_key``
    are both given:
        effective = base_price * (latest_month / base_month)
    using the row's ``fred_series``.  Missing base_month or any FRED
    failure falls back to the cited base price (reason in ``note``).
    """
    rows = load_price_table(csv_path)
    prices, meta = {}, {}
    for row in rows:
        name = row['solvent'].strip().lower()
        base = float(row['base_price_usd_per_kg'])
        base_year = int(row['base_year'])
        base_month = (row.get('base_month') or '').strip()
        effective = base
        escalated = False
        when = f"{base_year}-{int(base_month):02d}" if base_month \
            else str(base_year)
        note = f"cited base price ({when}, {row['region']})"
        series = (row.get('fred_series') or '').strip()

        if escalate and fred_api_key and series:
            try:
                now_date, idx_now = _fred_latest(series, fred_api_key)

                if base_month:
                    base_date, idx_base = _fred_month(
                        series, base_year, base_month, fred_api_key)
                    if idx_base is None:          # month not published
                        idx_base = _fred_year_average(
                            series, base_year, fred_api_key)
                        base_date = f"{base_year} avg"
                else:
                    idx_base = _fred_year_average(
                        series, base_year, fred_api_key)
                    base_date = f"{base_year} avg"

                if idx_now and idx_base:
                    factor = idx_now / idx_base
                    effective = base * factor
                    escalated = True
                    note = (f"escalated {base_date} -> {now_date} "
                            f"via FRED {series} (x{factor:.3f})")
                else:
                    note += "  [FRED returned no data - not escalated]"
            except urllib.error.HTTPError as exc:
                # Surface FRED's actual message (a bad key or bad series
                # ID both return HTTP 400 with an explanatory body).
                detail = f"HTTP {exc.code}"
                try:
                    body = exc.read().decode('utf-8', 'replace')
                    try:
                        detail += ": " + json.loads(body).get(
                            'error_message', body[:200])
                    except (ValueError, AttributeError):
                        detail += ": " + body[:200]
                except Exception:
                    pass
                note += f"  [escalation skipped: {detail}]"
            except urllib.error.URLError as exc:
                note += (f"  [escalation skipped: network error "
                         f"- {exc.reason}]")
            except Exception as exc:
                note += f"  [escalation skipped: {type(exc).__name__}]"

        prices[name] = effective
        meta[name] = dict(row)
        meta[name].update(base_price=base, effective_price=effective,
                          escalated=escalated, note=note)
    return prices, meta


# ── EIA: California industrial electricity (live monthly) ────────────────
def get_california_industrial_price(api_key):
    """Most recent monthly California industrial retail electricity price.

    Returns ``(price_usd_per_kwh, meta)``.  Raises ``RuntimeError`` if
    the key is missing, EIA returns an HTTP error, or no valid row is
    in the response.
    """
    if not api_key:
        raise RuntimeError(
            "EIA API key required: set the EIA_API_KEY environment "
            "variable.  Free key: "
            "https://www.eia.gov/opendata/register.php")

    # EIA v2 expects literal brackets in the query (data[0]=,
    # facets[..][]=, sort[0][...]=); build the query string manually so
    # urlencode does not percent-encode them.
    qs = (f"api_key={urllib.parse.quote(api_key, safe='')}"
          f"&frequency=monthly"
          f"&data[0]=price"
          f"&facets[stateid][]=CA"
          f"&facets[sectorid][]=IND"
          f"&sort[0][column]=period"
          f"&sort[0][direction]=desc"
          f"&offset=0&length=5")
    url = _EIA_RETAIL_URL + '?' + qs

    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            payload = json.load(resp)
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read().decode('utf-8', 'replace')[:300]
        except Exception:
            body = ''
        raise RuntimeError(
            f"EIA API returned HTTP {exc.code} -- {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"EIA API network error: {exc.reason}") from exc

    rows = payload.get('response', {}).get('data', [])
    for row in rows:
        try:
            price_cents = float(row['price'])
        except (KeyError, TypeError, ValueError):
            continue
        period = row.get('period', '?')
        meta = {
            'period': period,
            'price_cents_per_kwh': price_cents,
            'source': ('EIA Forms 826/861 -- California Industrial '
                       'sector average retail price'),
            'source_url': ('https://www.eia.gov/electricity/'
                           'monthly/update/end-use.php'),
            'note': (f"EIA California industrial retail electricity "
                     f"price for {period}"),
        }
        return price_cents / 100.0, meta

    raise RuntimeError(
        "EIA API returned no valid California industrial price rows.")


# ── BLS OEWS: California operator wage + optional ECI escalation ─────────
def get_california_operator_wage(api_key, fred_api_key=None):
    """Latest California state hourly mean wage for SOC 51-8091.

    Returns ``(wage_usd_per_hour, meta)``.  Raises ``RuntimeError`` if
    the BLS key is missing, the BLS API rejects the request, or no
    data is returned.

    If ``fred_api_key`` is supplied, the OEWS wage is escalated from
    its May reference quarter (Q2 of the OEWS year) to the latest
    quarterly ECIMANWAG observation.  Escalation is best-effort: any
    FRED failure returns the raw OEWS wage with the reason in ``note``.
    """
    if not api_key:
        raise RuntimeError(
            "BLS API key required: set the BLS_API_KEY environment "
            "variable.  Free key: "
            "https://data.bls.gov/registrationEngine/")

    body = json.dumps({
        "seriesid": [_CA_OPERATOR_HOURLY_SERIES],
        "registrationkey": api_key,
    }).encode("utf-8")
    req = urllib.request.Request(
        _BLS_URL, data=body,
        headers={"Content-Type": "application/json"})

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            payload = json.load(resp)
    except urllib.error.HTTPError as exc:
        try:
            err_body = exc.read().decode('utf-8', 'replace')[:300]
        except Exception:
            err_body = ''
        raise RuntimeError(
            f"BLS API returned HTTP {exc.code} -- {err_body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"BLS API network error: {exc.reason}") from exc

    if payload.get('status') != 'REQUEST_SUCCEEDED':
        msg = payload.get('message') or payload
        raise RuntimeError(f"BLS API error: {msg}")

    series = payload.get('Results', {}).get('series', [])
    if not series or not series[0].get('data'):
        raise RuntimeError(
            f"BLS API returned no data for series "
            f"{_CA_OPERATOR_HOURLY_SERIES}")

    base_wage = None
    period = None
    for obs in series[0]['data']:
        try:
            base_wage = float(obs['value'])
        except (KeyError, ValueError, TypeError):
            continue
        period = obs.get('year', '?')
        break
    if base_wage is None:
        raise RuntimeError("BLS API returned no valid wage observation.")

    meta = {
        'period': period,
        'base_wage': base_wage,
        'escalated': False,
        'soc': _SOC_CODE,
        'occupation': _OCCUPATION_NAME,
        'source': (f"BLS OEWS -- California state hourly mean wage for "
                   f"SOC {_SOC_CODE} ({_OCCUPATION_NAME})"),
        'source_url': 'https://www.bls.gov/oes/current/oes518091.htm',
        'note': (f"BLS OEWS California {_OCCUPATION_NAME} hourly mean "
                 f"wage, May {period} (not escalated)"),
    }

    # Optional ECI escalation: May falls in Q2 (Apr-Jun), so the base
    # ECI observation is Q2 of the OEWS reference year.
    if fred_api_key:
        try:
            base_date, eci_base = _fred_quarter(
                _ECI_SERIES, period, 2, fred_api_key)
            latest_date, eci_latest = _fred_latest(
                _ECI_SERIES, fred_api_key)
            if eci_base and eci_latest:
                factor = eci_latest / eci_base
                escalated_wage = base_wage * factor
                meta['escalated'] = True
                meta['escalation_factor'] = factor
                meta['eci_base_date'] = base_date
                meta['eci_latest_date'] = latest_date
                meta['note'] = (
                    f"BLS OEWS May {period} hourly mean wage "
                    f"(${base_wage:.2f}) escalated to {latest_date} "
                    f"via FRED {_ECI_SERIES} (x{factor:.3f})")
                return escalated_wage, meta
            else:
                meta['note'] += "  [ECI returned no data]"
        except urllib.error.HTTPError as exc:
            try:
                err_body = exc.read().decode('utf-8', 'replace')[:200]
            except Exception:
                err_body = ''
            meta['note'] += (f"  [ECI escalation skipped: HTTP "
                             f"{exc.code} -- {err_body}]")
        except urllib.error.URLError as exc:
            meta['note'] += (f"  [ECI escalation skipped: network "
                             f"error - {exc.reason}]")
        except Exception as exc:
            meta['note'] += (f"  [ECI escalation skipped: "
                             f"{type(exc).__name__}]")

    return base_wage, meta


if __name__ == '__main__':
    fred = os.environ.get('FRED_API_KEY')
    eia = os.environ.get('EIA_API_KEY')
    bls = os.environ.get('BLS_API_KEY')

    print("=== Solvent prices ===")
    print(f"FRED key {'found' if fred else 'NOT set - base prices only'}")
    prices, meta = get_solvent_prices(fred_api_key=fred,
                                      escalate=bool(fred))
    for s in prices:
        info = meta[s]
        print(f"  {s:11s} ${prices[s]:.4f}/kg   {info['note']}")
        print(f"              source: {info['source']}")

    print("\n=== California industrial electricity ===")
    if eia:
        price, m = get_california_industrial_price(eia)
        print(f"  ${price:.4f}/kWh   ({m['note']})")
        print(f"  source: {m['source']}")
    else:
        print("  Set EIA_API_KEY to test.")

    print("\n=== California operator wage (SOC 51-8091) ===")
    if bls:
        wage, m = get_california_operator_wage(bls, fred_api_key=fred)
        print(f"  ${wage:.2f}/hr   {m['note']}")
        print(f"  source: {m['source']}")
    else:
        print("  Set BLS_API_KEY to test.")
