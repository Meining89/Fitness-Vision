from utils.constant import *

def is_squatting_down(left_shoulder, right_shoulder, average_left_shoulder, average_right_shoulder, left_knee_angle, right_knee_angle, left_knee_average, right_knee_average):

    if is_shoulder_downwards(left_shoulder, right_shoulder, average_left_shoulder, average_right_shoulder) \
            and knee_bending(left_knee_angle, right_knee_angle, left_knee_average, right_knee_average):
        return True

    return False


def is_standing_up(left_shoulder, right_shoulder, average_left_shoulder, average_right_shoulder, left_knee_angle, right_knee_angle, left_knee_average, right_knee_average):

    if is_shoulder_upwards(left_shoulder, right_shoulder, average_left_shoulder, average_right_shoulder) \
            and knee_straightening(left_knee_angle, right_knee_angle, left_knee_average, right_knee_average):
        return True

    return False


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


def knee_bending(left_knee_angle, right_knee_angle, left_average, right_average, threshold=170):
    """
    Checks if the knees are bending during a squat
    :param left_knee_angle: angle of the left knee
    :param right_knee_angle: angle of the right knee
    :param left_average: moving average of the left knee angle
    :param right_average: moving average of the right knee angle
    :param threshold: angle threshold to determine if the knees are bending
    :return: True if the knees are bending, False otherwise
    """

    return (left_knee_angle < threshold
            and left_knee_angle < left_average
            and right_knee_angle < threshold
            and right_knee_angle < right_average)


def knee_straightening(left_knee_angle, right_knee_angle, left_average, right_average, threshold=90):
    """
    Checks if the knees are straightening during a squat
    :param left_knee_angle: angle of the left knee
    :param right_knee_angle: angle of the right knee
    :param left_average: moving average of the left knee angle
    :param right_average: moving average of the right knee angle
    :param threshold: angle threshold to determine if the knees are straightening
    :return: True if the knees are straightening, False otherwise
    """

    return (left_knee_angle > threshold
            and left_knee_angle > left_average
            and right_knee_angle > threshold
            and right_knee_angle > right_average)
