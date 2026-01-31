import json
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

rows = [
    {
        "UWI": "MOCK-OZALPHA-1",
        "CASING_TYPE": "Conductor",
        "OUTER_DIAMETER": '14"',
        "CASING_TOP": "7.7",
        "CASING_BOTTOM": "6.7",
    },
    {
        "UWI": "MOCK-OZALPHA-1",
        "CASING_TYPE": "36 ppf J-55 BTC",
        "OUTER_DIAMETER": '9 5/8"',
        "CASING_TOP": "507.5",
        "CASING_BOTTOM": "505",
    },
    {
        "UWI": "MOCK-OZALPHA-1",
        "CASING_TYPE": "13.5 ppf L-80 Tenaris Blue",
        "OUTER_DIAMETER": '4 1/2"',
        "CASING_TOP": "1250.3",
        "CASING_BOTTOM": "1240",
    }
]

resp = client.post('/check-existence', data={
    'data': json.dumps(rows),
    'table_name': 'WCR_CASING'
})

print('Status:', resp.status_code)
print(resp.json())
