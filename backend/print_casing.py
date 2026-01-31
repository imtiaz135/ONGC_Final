from database import engine
import pandas as pd

df = pd.read_sql('SELECT * FROM WCR_CASING', engine)
print(df.to_dict('records'))
