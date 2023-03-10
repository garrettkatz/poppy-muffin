import pickle as pk

with open('tmp.pkl', "rb") as f:
    success, bufs, time_elapsed = pk.load(f, encoding='latin1')

import matplotlib.pyplot as pt

pt.ion()
pt.show()

# for binsize in range(1,11):
#     for t in range(len(bufs['images'])):
#         img = bufs['images'][t][::binsize,::binsize,[2,1,0]].astype(float)/255.
#         pt.imshow(img)
#         pt.pause(0.01)

#     input('.')
#     pt.close()

for t in range(len(bufs['images'])):
    img = bufs['images'][t][:,:,[2,1,0]].astype(float)/255.
    pt.imshow(img)
    input('.')
pt.close()


