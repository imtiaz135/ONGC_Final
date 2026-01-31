from database import engine
import pandas as pd
from main import build_casing_signature, casing_rows_match

# Load DB rows
existing_df = pd.read_sql('SELECT * FROM WCR_CASING', engine)
existing = existing_df.to_dict('records')

print('DB rows:')
for r in existing:
    print(r)

db_sigs = [build_casing_signature(r) for r in existing]
print('\nDB signatures:')
for s in db_sigs:
    print(s)

# Input rows mimicking extracted values
input_rows = [
    {"UWI":"MOCK-OZALPHA-1","CASING_TYPE":"Conductor","OUTER_DIAMETER":"14\"","CASING_TOP":"7.7","CASING_BOTTOM":"6.7"},
    {"UWI":"MOCK-OZALPHA-1","CASING_TYPE":"36 ppf J-55 BTC","OUTER_DIAMETER":"9 5/8\"","CASING_TOP":"507.5","CASING_BOTTOM":"505"},
    {"UWI":"MOCK-OZALPHA-1","CASING_TYPE":"13.5 ppf L-80 Tenaris Blue","OUTER_DIAMETER":"4 1/2\"","CASING_TOP":"1250.3","CASING_BOTTOM":"1240"},
]

print('\nInput signatures and matching:')
for inp in input_rows:
    sig = build_casing_signature(inp)
    print('INPUT SIG:', sig)
    for i, dbs in enumerate(db_sigs):
        ok = casing_rows_match(sig, dbs)
        print('  matches DB row', i, ok)
