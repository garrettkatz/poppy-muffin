import matplotlib.pyplot as pt
import pickle as pk

with open("getcam.pkl","rb") as f: rgb, depth, segment = pk.load(f)

pt.imshow(rgb)
pt.show()

