from utils.constant import *
from utils.draw_display import *


def process_shallow(frame, counter, knee_obj):

    min_knee_angle = min(knee_obj.left_angle, knee_obj.right_angle)

    frame_height, frame_width, _ = frame.shape

    # display knee_angle at knee_loc
    left_knee_pixel_x = int(knee_obj.left.x * frame_width)
    left_knee_pixel_y = int(knee_obj.left.y * frame_height)
    knee_loc = (left_knee_pixel_x + 10, left_knee_pixel_y)
    knee_angle = min(knee_obj.left_angle, knee_obj.right_angle)
    knee_angle_text = f"{knee_angle:.2f} degrees"
    draw_text(frame, knee_loc, knee_angle_text)
    _, knee_text_height = cv2.getTextSize(knee_angle_text, cv2.FONT_HERSHEY_SIMPLEX, 2, thickness=2)[0]

    if not counter.going_up and min_knee_angle > KNEE_ANGLE_DEPTH:
        text_to_display = "Go lower!"
        draw_text(frame, (knee_loc[0], knee_loc[1] + knee_text_height + 20), text_to_display, font_scale=2,
                  color=(0, 0, 255))
        return True

    return False

def is_squatting_down(shoulder_obj, knee_obj, threshold=170):

    left_shoulder = shoulder_obj.left.y
    right_shoulder = shoulder_obj.right.y
    average_left_shoulder = shoulder_obj.avg_val_left
    average_right_shoulder = shoulder_obj.avg_val_right

    left_knee_angle = knee_obj.left_angle
    right_knee_angle = knee_obj.right_angle
    left_knee_average = knee_obj.avg_angle_left
    right_knee_average = knee_obj.avg_angle_right

    if is_shoulder_downwards(left_shoulder, right_shoulder, average_left_shoulder, average_right_shoulder) \
            and knee_bending(left_knee_angle, right_knee_angle, left_knee_average, right_knee_average, threshold):
        return True

    return False


def is_standing_up(shoulder_obj, knee_obj):

    left_shoulder = shoulder_obj.left.y
    right_shoulder = shoulder_obj.right.y
    average_left_shoulder = shoulder_obj.avg_val_left
    average_right_shoulder = shoulder_obj.avg_val_right

    left_knee_angle = knee_obj.left_angle
    right_knee_angle = knee_obj.right_angle
    left_knee_average = knee_obj.avg_angle_left
    right_knee_average = knee_obj.avg_angle_right

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


# OLDER CODE
def is_squatting_down_old(left_shoulder, right_shoulder, average_left_shoulder, average_right_shoulder, left_knee_angle, right_knee_angle, left_knee_average, right_knee_average):

    if is_shoulder_downwards(left_shoulder, right_shoulder, average_left_shoulder, average_right_shoulder) \
            and knee_bending(left_knee_angle, right_knee_angle, left_knee_average, right_knee_average):
        return True

    return False


def is_standing_up_old(left_shoulder, right_shoulder, average_left_shoulder, average_right_shoulder, left_knee_angle, right_knee_angle, left_knee_average, right_knee_average):

    if is_shoulder_upwards(left_shoulder, right_shoulder, average_left_shoulder, average_right_shoulder) \
            and knee_straightening(left_knee_angle, right_knee_angle, left_knee_average, right_knee_average):
        return True

    return False

