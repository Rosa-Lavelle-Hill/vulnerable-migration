import plotly.graph_objects as go
import math
import networkx as nx

from Functions.Database import select, create_table
from decimal import Decimal

f = open("SQL/Queries/Create_Tables/Ward_rural_urban_class_geom.txt")
query1 = f.read()

f = open("SQL/Queries/Select/Ward_rural_urban_class_geom.txt")
query2 = f.read()

if __name__ == '__main__':
    create_table(query1)
    df = select(query2)




print('done!')