from psycopmlutils.loaders.load_ids import LoadIDs

import pandas as pd

def load_split(split):
    return LoadIDs.load(split)["dw_ek_borger"]

train =  load_split("train")
val = load_split("val")
test = load_split("test")


val[val.isin(train)]
test[test.isin(train)]

