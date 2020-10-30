import numpy as np
import cv2
from utils.constants import Vertices, HoughLineConst, CannyConst, GaussianBlurConst
from units.line import Line


def region_of_interest(img, vertices):
    """
    Applied an image mask.
    Only keeps the iamge defined by the vertices formed from the scooter.
    The rest of image is set to black.
    :param img:
    :param vertices:
    :return: masked image, mask
    """
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(img, mask)
    return mask, masked


def add_weight(img, init_img, alpha=0.8, beta=1., lambd=0.):
    # Create a "color" binary image to combine with line image
    img = np.uint8(img)
    # Draw the lines on the initial image
    combo = cv2.addWeighted(init_img, alpha, img, beta, lambd)
    return combo


def compute_lane_from_candidates(line_candidates, img_shape):
    """
    Compute lines that approximate the position of both road lanes.

    :param line_candidates: lines from hough transform
    :param img_shape: shape of image to which hough transform was applied
    :return: lines that approximate left and right lane position
    """

    # separate candidate lines according to their slope
    pos_lines = [line for line in line_candidates if line.slope > 0]
    neg_lines = [line for line in line_candidates if line.slope < 0]

    # interpolate biases and slopes to compute equation of line that approximates left lane
    # median is employed to filter outliers
    neg_bias = np.median([line.bias for line in neg_lines]).astype(int)
    neg_slope = np.median([line.slope for line in neg_lines])
    x1, y1 = 0, neg_bias
    x2, y2 = -np.int32(np.round(neg_bias / neg_slope)), 0
    left_lane = Line(x1, y1, x2, y2)

    # interpolate biases and slopes to compute equation of line that approximates right lane
    # median is employed to filter outliers
    lane_right_bias = np.median([line.bias for line in pos_lines]).astype(int)
    lane_right_slope = np.median([line.slope for line in pos_lines])
    x1, y1 = 0, lane_right_bias
    x2, y2 = np.int32(np.round((img_shape[0] - lane_right_bias) / lane_right_slope)), img_shape[0]
    right_lane = Line(x1, y1, x2, y2)

    return left_lane, right_lane


def find_lane(image, infer_lines=True):
    # convert image to graysclae
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # gaussian blur
    blur_gray = cv2.GaussianBlur(gray, GaussianBlurConst.kernel, 0)
    # canny edge detection
    edges = cv2.Canny(blur_gray, CannyConst.threshold1, CannyConst.threshold2)

    # Hough transform
    lines = cv2.HoughLinesP(edges, HoughLineConst.rho, HoughLineConst.theta,
                            HoughLineConst.threshold, np.array([]),
                            HoughLineConst.min_line_len, HoughLineConst.max_line_gap)

    # creating a blank to draw lines on
    line_image = np.copy(image) * 0
    try:
        detected_lines = [Line(line[0][0], line[0][1], line[0][2], line[0][3]) for line in lines]
        # if 'infer_lines' infer the two lane lines
        if infer_lines:
            candidate_lines = []
            for line in detected_lines:
                # consider only lines with slope between 30 and 60 degrees
                if 0.5 <= np.abs(line.slope) <= 2:
                    candidate_lines.append(line)
            # interpolate lines candidates to find both lanes
            lane_lines = compute_lane_from_candidates(candidate_lines, gray.shape)
        else:
            # if not solid_lines, just return the hough transform output
            lane_lines = detected_lines

        for line in lane_lines:
            line.draw(line_image)

    except:
        pass
    _, masked_image = region_of_interest(line_image, Vertices.common)
    combo = add_weight(masked_image, image, 0.8, 1., 0.)
    return combo
