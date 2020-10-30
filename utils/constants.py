import numpy as np


class CannyConst:
    threshold1=50
    threshold2=150


class GaussianBlurConst:
    kernel = (5, 5)


class Vertices:
    scooter = np.array([[[10, 500], [10, 300], [300, 200],
                         [500, 200], [800, 300], [800, 500]]])
    common = np.array([[(0, 600), (0, 250),
                        (800, 250), (800, 600)]],
                      dtype=np.int32)


class HoughLineConst:
    rho = 1
    theta = np.pi / 180
    threshold = 160
    min_line_len = 15
    max_line_gap = 5