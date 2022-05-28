from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import string
import numpy as np
from TrueSightEngine import SearchEngine as TS
from TrueSightEngine import TimeExecution
from datetime import datetime

# buildder = {}

# TS.addDataToDictionary({'berita': 'berita ke1', 'isi': 'random1'}, buildder)
# TS.addDataToDictionary({'berita': 'berita ke2', 'isi': 'random2'}, buildder)
# TS.addDataToDictionary({'berita': 'berita ke3', 'isi': 'random3'}, buildder)

# print(buildder)

file_csv = input("Lokasi csv: ")
reader = pd.read_csv(file_csv)

header = input("Lihat Kolom Header: ")
cari = input('Cari kalimat: ')

data = reader.to_dict()

TE = TimeExecution()
TE.init()

result = TS.search_from_dict(
    keywords=cari,
    data=data,
    lookupHeader=[header]
)
print("Total time spend")
TE.end()

THRESHOLD = 0.1

for i, (v, x) in enumerate(result):
    if v > THRESHOLD:
        print(str(i) + f"({str(v)})", '=>', x['berita'])

if len(result) == 0:
    print("Not found")
