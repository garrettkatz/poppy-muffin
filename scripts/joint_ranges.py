import numpy as np
import pickle as pk
import matplotlib.pyplot as pt
from record_angles import show_angles

if __name__ == "__main__":
    with open("joint_range_recording.pkl","r") as f: angles = pk.load(f)
    # show_angles(angles)

    angles = {
        n: np.array([a[n] for a in angles])
        for n in angles[0].keys()}

    # print(angles)
    # for n in angles.keys():
    for n in ["r_shoulder_y","r_hip_z","l_hip_z","l_shoulder_y"]:
    
        print("motor %s: [%.1f, %.1f]" % (n, angles[n].min(), angles[n].max()))

        dsct = np.flatnonzero(np.fabs(angles[n][1:] - angles[n][:-1]) > 300)
        print("  discontinuities:")
        print(dsct)
        for d in dsct:
            print("%.1f ~ %.1f" % (angles[n][d], angles[n][d+1]))
        
        smoothed = np.copy(angles[n])
        for d in dsct:
            smoothed[d+1:] += (-1)**(smoothed[d+1] > smoothed[d]) * 360

        print("  smoothed: [%.1f, %.1f]" % (smoothed.min(), smoothed.max()))
        print("%.1f" % np.fabs(smoothed[1:]-smoothed[:-1]).max())
        pt.plot(angles[n])
        pt.plot(smoothed)
        pt.title("%s [%.1f to %.1f] ~ [%.1f to %.1f]" % (n, angles[n].min(), angles[n].max(), smoothed.min(), smoothed.max()))
        pt.legend(["raw","smoothed"])
        pt.show()


