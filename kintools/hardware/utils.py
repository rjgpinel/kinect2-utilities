import yaml
import numpy as np
import glob
import os


def load_calibration_file(filename):
    params = yaml.load(open(filename, "r"), Loader=yaml.FullLoader)
    for k, v in params.items():
        params[k] = np.array(v)
        if k == "cameraMatrix":
            params[k] = params[k].reshape(3, 3)
        elif k == "projection":
            params[k] = params[k].reshape(4, 4)
    return params


def load_calibration(calibration_folder):
    files = glob.glob(os.path.join(calibration_folder, "*.yaml"))
    calib_type = ["color", "depth", "ir", "pose"]
    calibration = {}
    for fn in files:
        for calib in calib_type:
            if calib in fn:
                calibration[calib] = load_calibration_file(fn)
                break
    return calibration
