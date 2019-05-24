#from pypot.creatures import PoppyHumanoid as PH

#p = PH()

if True:
    from pypot.sensor.camera.opencvcam import OpenCVCamera
    import matplotlib.pyplot as plt

    c = OpenCVCamera("testcam",0,10)

    print(c)
    print(c.frame.shape)
    print(c.frame.dtype)
    plt.imshow(c.frame[:,:,[2,1,0]])
    plt.show()

    c.close()

