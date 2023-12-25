from utils.constant import *


def is_shoulder_upwards(left_shoulder, right_shoulder, average_left_shoulder, average_right_shoulder):
    """
    Checks if the shoulders are above the average shoulder position. Note: In image coordinates, y increases downwards
    :param left_shoulder: y coordinate of the left shoulder landmark
    :param right_shoulder: y coordinate of the right shoulder landmark
    :param average_left_shoulder: average y coordinate of the left shoulder landmark
    :param average_right_shoulder: average y coordinate of the right shoulder landmark
    :return:
    """

    # left shoulder y less than average means left shoulder is higher than average
    if left_shoulder < average_left_shoulder - MOVEMENT_THR \
            and right_shoulder < average_right_shoulder - MOVEMENT_THR:
        return True

    return False


def is_shoulder_downwards(left_shoulder, right_shoulder, average_left_shoulder, average_right_shoulder):
    """
    Checks if the shoulders are below the average shoulder position. Note: In image coordinates, y increases downwards
    :param left_shoulder: y coordinate of the left shoulder landmark
    :param right_shoulder: y coordinate of the right shoulder landmark
    :param average_left_shoulder: average y coordinate of the left shoulder landmark
    :param average_right_shoulder: average y coordinate of the right shoulder landmark
    :return:
    """

    # left shoulder y more than average means left shoulder is lower than average
    if left_shoulder > average_left_shoulder + MOVEMENT_THR \
            and right_shoulder > average_right_shoulder + MOVEMENT_THR:
        return True

    return False
