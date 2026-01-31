from sqlalchemy import text
from database import engine


def normalize_numeric_value(val):
    if val is None:
        return None
    s = str(val).strip()
    for unit in ['"', "'", 'ppf', 'm', 'mm', 'in', 'ft', 'psi', 'PSI', 'kg', 'lb']:
        s = s.replace(unit, '')
    try:
        return float(s)
    except Exception:
        return s.upper()


def build_casing_signature(row):
    return {
        'UWI': str(row.get('UWI', '')).strip().upper() if row.get('UWI') else None,
        'CASING_TYPE': str(row.get('CASING_TYPE', '')).strip().upper() if row.get('CASING_TYPE') else None,
        'CASING_TOP': normalize_numeric_value(row.get('CASING_TOP')),
        'CASING_BOTTOM': normalize_numeric_value(row.get('CASING_BOTTOM')),
        'OUTER_DIAMETER': normalize_numeric_value(row.get('OUTER_DIAMETER')),
    }


def casing_rows_match(ex_sig, db_sig, tolerance=0.5):
    if not ex_sig.get('UWI') or not db_sig.get('UWI'):
        return False
    if ex_sig['UWI'] != db_sig['UWI']:
        return False
    match_count = 0
    fields = ['CASING_TYPE', 'CASING_TOP', 'CASING_BOTTOM', 'OUTER_DIAMETER']
    for f in fields:
        ev = ex_sig.get(f)
        dv = db_sig.get(f)
        if ev is None or dv is None:
            continue
        if isinstance(ev, float) and isinstance(dv, float):
            if abs(ev - dv) <= tolerance:
                match_count += 1
        else:
            if str(ev) == str(dv):
                match_count += 1
    return match_count >= 4


def insert_if_missing(rows):
    inserted = 0
    with engine.begin() as conn:
        # Fetch existing rows for this UWI
        uwis = set([r['UWI'] for r in rows if r.get('UWI')])
        existing = []
        for u in uwis:
            res = conn.execute(text('SELECT * FROM WCR_CASING WHERE UWI = :u'), {'u': u}).mappings().all()
            for r in res:
                existing.append(dict(r))

        db_sigs = [build_casing_signature(r) for r in existing]

        for row in rows:
            sig = build_casing_signature(row)
            found = False
            for dbs in db_sigs:
                if casing_rows_match(sig, dbs):
                    found = True
                    break
            if found:
                print(f"SKIP (exists): {row}")
                continue

            # Insert
            insert_sql = text('''
                INSERT INTO WCR_CASING (
                    UWI, CASING_TYPE, CASING_TOP, CASING_BOTTOM, OUTER_DIAMETER, WEIGHT, STEEL_GRADE, REMARKS, PAGE_NUMBERS
                ) VALUES (
                    :uwi, :ct, :top, :bot, :od, :weight, :grade, :remarks, :page
                )
            ''')
            params = {
                'uwi': row.get('UWI'),
                'ct': row.get('CASING_TYPE'),
                'top': row.get('CASING_TOP'),
                'bot': row.get('CASING_BOTTOM'),
                'od': row.get('OUTER_DIAMETER'),
                'weight': row.get('WEIGHT'),
                'grade': row.get('STEEL_GRADE'),
                'remarks': row.get('REMARKS'),
                'page': row.get('PAGE_NUMBERS') or 'Casing'
            }
            conn.execute(insert_sql, params)
            inserted += 1
            # Add to db_sigs so subsequent rows see it
            db_sigs.append(sig)

    print(f"Inserted {inserted} new casing rows.")


if __name__ == '__main__':
    # Rows from the attached table
    rows = [
        {
            'UWI': 'MOCK-OZALPHA-1',
            'CASING_TYPE': 'Conductor',
            'CASING_TOP': 7.7,
            'CASING_BOTTOM': 6.7,
            'OUTER_DIAMETER': 14,
            'REMARKS': 'Hole size 17 1/2"',
            'PAGE_NUMBERS': 'Casing'
        },
        {
            'UWI': 'MOCK-OZALPHA-1',
            'CASING_TYPE': '36 ppf J-55 BTC',
            'CASING_TOP': 507.5,
            'CASING_BOTTOM': 505,
            'OUTER_DIAMETER': 9.625,
            'WEIGHT': '36 ppf',
            'STEEL_GRADE': 'J-55',
            'REMARKS': 'BTC; LOT test 11.4 ppg at 511 m on 08.04.2014',
            'PAGE_NUMBERS': 'Casing'
        },
        {
            'UWI': 'MOCK-OZALPHA-1',
            'CASING_TYPE': '13.5 ppf L-80 Tenaris Blue',
            'CASING_TOP': 1250.3,
            'CASING_BOTTOM': 1240,
            'OUTER_DIAMETER': 4.5,
            'WEIGHT': '13.5 ppf',
            'STEEL_GRADE': 'L-80',
            'REMARKS': 'Tenaris Blue; Hole size 7 7/8"',
            'PAGE_NUMBERS': 'Casing'
        }
    ]

    insert_if_missing(rows)
