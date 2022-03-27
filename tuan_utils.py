import math

import cv2
import numpy as np


def get_max_x_rect(rect_line):
    return max(rect_line[1], key=lambda tx: tx[1])[0]


def is_confirm_by_shifted(a):
    if a < 10:
        return True
    else:
        return False


def is_valid_angle(a):
    return 75 < a < 100


def minimum_bounding_rectangle(points):
    min_rect = cv2.minAreaRect(points)
    box = cv2.boxPoints(min_rect)
    box = np.intp(box)
    return box


def sort_rec(rec):
    sorted_r = sorted(rec, key=lambda x: x[0])
    if sorted_r[0][1] > sorted_r[1][1]:
        temp = sorted_r[1][1]
        sorted_r[1][1] = sorted_r[0][1]
        sorted_r[0][1] = temp
    if sorted_r[2][1] < sorted_r[3][1]:
        temp = sorted_r[2][1]
        sorted_r[2][1] = sorted_r[3][1]
        sorted_r[3][1] = temp
    return sorted_r


def calculate_angle_rectangles(p_rect, c_rect):
    ys2 = [(p_rect[0][1] + p_rect[1][1]) / 2,
           (c_rect[2][1] + c_rect[3][1]) / 2]
    xs2 = [(p_rect[0][0] + p_rect[1][0]) / 2,
           (c_rect[2][0] + c_rect[3][0]) / 2]

    ys = [(p_rect[3][1] + p_rect[2][1]) / 2,
          (c_rect[2][1] + c_rect[3][1]) / 2]
    xs = [(p_rect[3][0] + p_rect[2][0]) / 2,
          (c_rect[2][0] + c_rect[3][0]) / 2]

    middle_line = ((xs2[0], ys2[0]), (xs2[1], ys2[1]))
    left_line = (c_rect[0], c_rect[1])
    right_line = (c_rect[2], c_rect[3])

    intersect_result = [line_intersection(middle_line, left_line), \
                        line_intersection(middle_line, right_line)]

    if intersect_result:
        intersect_angle_l, intersect_angle_r, intersect_angle_u, intersect_angle_d = get_intersect_angle(c_rect, xs2,
                                                                                                         ys2)
        intersect_angle_l2, intersect_angle_r2, _, _ = get_intersect_angle(c_rect, xs, ys)

        current_range_y = set(range(p_rect[0][1].astype(int), p_rect[1][1].astype(int)))
        range1_y = set(range(c_rect[0][1].astype(int), c_rect[1][1].astype(int)))
        if min(len(range1_y), len(current_range_y)) == 0:
            y_overlapped_before_resize_percentage = 0
        else:
            y_overlapped_before_resize_percentage = len(current_range_y.intersection(range1_y)) / min(len(range1_y),
                                                                                                      len(
                                                                                                          current_range_y))

        # resize for height
        if (p_rect[1][1] - p_rect[0][1] > 25):
            p_rect[0][1] = p_rect[0][1] + 10
            p_rect[1][1] = p_rect[1][1] - 10
            p_rect[3][1] = p_rect[3][1] + 10
            p_rect[2][1] = p_rect[2][1] - 10

        if (c_rect[1][1] - c_rect[0][1] > 25):
            c_rect[0][1] = c_rect[0][1] + 10
            c_rect[1][1] = c_rect[1][1] - 10
            c_rect[3][1] = c_rect[3][1] + 10
            c_rect[2][1] = c_rect[2][1] - 10

        # resize to 50 length
        if p_rect[3][0] - p_rect[0][0] > 50:
            p_rect[3][0] = p_rect[0][0] + 50
            p_rect[2][0] = p_rect[0][0] + 50

        if c_rect[3][0] - c_rect[0][0] > 50:
            c_rect[3][0] = c_rect[0][0] + 50
            c_rect[2][0] = c_rect[0][0] + 50

        delta_x = (c_rect[0][0] - p_rect[0][0])
        prev_upper_half_length = (p_rect[1][0] - p_rect[0][0]) / 2

        shifted_rect_0 = (c_rect[0][0] - delta_x - prev_upper_half_length, c_rect[0][1])
        shifted_rect_1 = (c_rect[1][0] - delta_x - prev_upper_half_length, c_rect[1][1])
        shifted_rect_2 = (c_rect[2][0] - delta_x - prev_upper_half_length, c_rect[2][1])
        shifted_rect_3 = (c_rect[3][0] - delta_x - prev_upper_half_length, c_rect[3][1])

        shifted_p1 = (p_rect[0][0] + p_rect[1][0]) / 2, (p_rect[0][1] + p_rect[1][1]) / 2,
        shifted_p2 = (shifted_rect_2[0] + shifted_rect_3[0]) / 2, (shifted_rect_2[1] + shifted_rect_3[1]) / 2

        shifted_intersect_angle_l_3 = get_angle((shifted_rect_0, shifted_rect_1), (shifted_p1, shifted_p2))
        shifted_intersect_angle_r_3 = get_angle((shifted_rect_3, shifted_rect_2), (shifted_p1, shifted_p2))

        intersect_angle_l_2 = get_angle((c_rect[0], c_rect[1]), ((xs[0], ys[0]), (xs[1], ys[1])))
        intersect_angle_r_2 = get_angle((c_rect[3], c_rect[2]), ((xs[0], ys[0]), (xs[1], ys[1])))
        return intersect_angle_l, intersect_angle_r, intersect_angle_u, intersect_angle_d, \
               intersect_angle_l_2, intersect_angle_r_2, shifted_intersect_angle_l_3, shifted_intersect_angle_r_3, y_overlapped_before_resize_percentage, intersect_result
    else:
        return None


def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        return -1, -1

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return [x, y]


def get_angle(l1, l2):
    if l1[1][0] == l1[0][0]:
        m1 = l1[1][1] - l1[0][1]
    else:
        m1 = (l1[1][1] - l1[0][1]) / (l1[1][0] - l1[0][0])

    if l2[1][0] == l2[0][0]:
        m2 = l2[1][1] - l2[0][1]
    else:
        m2 = (l2[1][1] - l2[0][1]) / (l2[1][0] - l2[0][0])

    angle_rad = abs(math.atan(m1) - math.atan(m2))
    angle_deg = angle_rad * 180 / math.pi
    return angle_deg


def get_intersect_angle(c_rect, xs2, ys2):
    intersect_angle_l = get_angle((c_rect[0], c_rect[1]), ((xs2[0], ys2[0]), (xs2[1], ys2[1])))
    intersect_angle_r = get_angle((c_rect[3], c_rect[2]), ((xs2[0], ys2[0]), (xs2[1], ys2[1])))
    intersect_angle_u = get_angle((c_rect[0], c_rect[3]), ((xs2[0], ys2[0]), (xs2[1], ys2[1])))
    intersect_angle_d = get_angle((c_rect[1], c_rect[2]), ((xs2[0], ys2[0]), (xs2[1], ys2[1])))
    return intersect_angle_l, intersect_angle_r, intersect_angle_u, intersect_angle_d
