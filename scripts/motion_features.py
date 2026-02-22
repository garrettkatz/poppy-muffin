import matplotlib.pyplot as pt
import numpy as np
from scipy.signal import correlate
# from scipy.fft import fft2, ifft2, fftshift
# from scipy.ndimage import rotate, zoom
# from skimage.transform import warp_polar

def get_motion_features(img1, img2):

    # convert to intensity
    img1 = img1.mean(axis=2)
    img2 = img2.mean(axis=2)
    h, w = img1.shape

    # mean-center pixel values  so zero-padding does not bias towards zero translation
    img1 -= img1.mean()
    img2 -= img2.mean()

    # TODO: log-polar rotation/scale registration first, something like 
    
    # # --- STAGE 1: Rotation and Scale (Translation Invariant) ---
    # f1_mag = fftshift(np.abs(fft2(img1)))
    # f2_mag = fftshift(np.abs(fft2(img2)))
    
    # max_radius = np.sqrt((h/2)**2 + (w/2)**2)
    
    # # Map Magnitude Spectra to Log-Polar
    # lp1 = warp_polar(f1_mag, center=(h/2, w/2), radius=max_radius, scaling='log')
    # lp2 = warp_polar(f2_mag, center=(h/2, w/2), radius=max_radius, scaling='log')
    
    # # Phase Correlation on Log-Polar gives Rotation (v) and Scale (h)
    # res_rs = np.abs(ifft2(fft2(lp1) * fft2(lp2).conj() / (np.abs(fft2(lp1) * fft2(lp2).conj()) + 1e-9)))
    # dv, dh = np.unravel_index(np.argmax(res_rs), res_rs.shape)
    
    # # Convert shifts to actual units
    # angle = (dv * 360.0) / lp1.shape[0]  # Degrees
    # scale = np.exp(dh * np.log(max_radius) / lp1.shape[1]) # Scale factor
    
    # # --- STAGE 2: Translation (Rotation & Scale Invariant) ---
    # # To get translation, we must first "undo" the rotation/scale on img2
    # # Note: For a "rough feature", you could skip the undoing and just 
    # # run phase correlation on the original images, but it will be noisy.
    # img2_corrected = rotate(img2, -angle, reshape=False)
    # # (Optional: apply zoom(img2_corrected, 1/scale) here for high precision)

    ### Just do translation for now
    angle, scale = 0, 1
    img2_corrected = img2
    
    correlation = correlate(img1, img2, mode='same', method='auto')

    # # hard max
    # my, mx = np.unravel_index(np.argmax(correlation), correlation.shape)

    # soft max
    weights = correlation - correlation.min()
    weights = weights / weights.sum()
    mx = (weights.sum(axis=0) * np.arange(w)).sum()
    my = (weights.sum(axis=1) * np.arange(h)).sum()

    # offset from center
    dx, dy = mx - (w / 2), my - (h / 2)

    # Final Motion Signature
    return np.array([dx, dy, angle, scale]), correlation

if __name__ == "__main__":

    import pickle as pk
    from stitch_preanalysis import get_run_filepaths

    run_filepaths = get_run_filepaths()

    traj_file, buf_file, stdev, num_success = run_filepaths[2]

    ## load the data
    with open(buf_file, "rb") as f:
        (buffers, elapsed, waytime_points, all_motor_names) = pk.load(f, encoding='latin1')

    imgs = [frame[:,:,[2,1,0]].astype(float)/255. for frame in buffers["images"] if frame is not None]
    timestamps = [t for (t,frame) in zip(elapsed, buffers["images"]) if frame is not None]
    print(f"{len(imgs)} imgs")

    dxs, dys = [], []
    # pt.ion()
    # pt.show()
    for t in range(len(imgs)-1):

        img0 = imgs[t]
        img1 = imgs[t+1]
    
        (dx, dy, angle, scale), correlation = get_motion_features(img0, img1)
        dxs.append(dx)
        dys.append(dy)
        print(f"{t=} of {len(imgs)}, {dx=}, {dy=}, {angle=}, {scale=}")
    
        # pt.subplot(1,3,1)
        # pt.imshow(img0)
        # pt.subplot(1,3,2)
        # pt.imshow(img1)
        # pt.subplot(1,3,3)
        # pt.imshow(correlation)
        # pt.show()
        # pt.pause(0.1)
        # input("...")

    # pt.close()

    print(buf_file)

    pt.plot(timestamps[1:], dxs, label="dx")
    pt.plot(timestamps[1:], dys, label="dy")
    pt.xlabel("Time (sec)")
    pt.ylabel("displacement")
    pt.legend()
    pt.show()

