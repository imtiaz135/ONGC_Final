import asyncio
import json
from main import check_existence


async def run_test():
    # Simulate extracted CASING rows (with unit variations and strings)
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

    data_str = json.dumps(rows)
    # Call the async endpoint function directly
    result = await check_existence(data=data_str, table_name="WCR_CASING")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(run_test())
