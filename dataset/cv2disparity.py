import cv2
import numpy as np
import matplotlib.pyplot as plt


def compute_disparity_uncertainty(iml, imr, ret_left=True, resize=False, mindisp=0):
    """
    return disp, uncertainty map
    the disp will always be positive
    """
    hei, wid, _ = iml.shape
    if resize:
        iml = cv2.resize(iml, None, None, 0.5, 0.5)
        imr = cv2.resize(imr, None, None, 0.5, 0.5)

    if not ret_left:
        iml_tmp = iml
        iml = cv2.flip(imr, 1)
        imr = cv2.flip(iml_tmp, 1)
    # figuring out parameters
    numdisp = int(wid * 0.07)
    if numdisp % 16 != 0:
        numdisp += int(16 - (numdisp % 16))
    wsize = 3

    left_matcher = cv2.StereoSGBM_create(
        minDisparity=mindisp,
        numDisparities=numdisp,
        blockSize=wsize,
        P1=24*wsize*wsize,
        P2=96*wsize*wsize,
        preFilterCap=65
    )

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(left_matcher)
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
    iml_formatch, imr_formatch = iml, imr
    displ = left_matcher.compute(iml_formatch, imr_formatch)
    dispr = right_matcher.compute(imr_formatch, iml_formatch)
    wls_filter.setLambda(8000)
    wls_filter.setSigmaColor(1.)
    disp = wls_filter.filter(displ, iml, None, dispr)

    uncertainty = wls_filter.getConfidenceMap()
    uncertainty = (uncertainty / 255).astype(np.float32)

    disp = (disp / 16).astype(np.float32)
    disp = np.where(disp < 0, 0, disp)
    # this is not needed because uncertainty already has 0 value for mask
    # mask = disp < 0
    # disp[mask] = 0
    # uncertainty[mask] = 0

    if not ret_left:
        disp = cv2.flip(disp, 1)
        uncertainty = cv2.flip(uncertainty, 1)
    if resize:
        disp = cv2.resize(disp, None, None, 2, 2)
        uncertainty = cv2.resize(uncertainty, None, None, 2, 2)
    return disp, uncertainty
