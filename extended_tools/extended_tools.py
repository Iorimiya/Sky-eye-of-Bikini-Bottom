import cv2
import numpy as np
import pandas as pd
import deeplabcut as DLC
import subprocess
import os


def create_label_video_with_rotational_view(
        config_file_path,
        video_folder_path,
        result_names,
        start_frame,
        frame_duration=300,
        start_view_horizontal=0,
        start_view_vertical=60,
        source_video_type='mp4',
        xlim=[None, None],
        ylim=[None, None],
        zlim=[None, None]):
    for result_name in result_names:
        if os.path.exists(os.path.join(video_folder_path, result_name + '.mpg')):
            os.remove(os.path.join(video_folder_path, result_name + '.mpg'))
        temp_dir = os.path.join(video_folder_path, "temp_" + result_name)
        if os.path.exists(temp_dir):
            for root, dirs, files in os.walk(temp_dir, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))

    for index in range(0, frame_duration):
        frame = start_frame + index
        view_vertical = start_view_vertical
        view_horizontal = start_view_horizontal + index
        DLC.create_labeled_video_3d(config=config_file_path, path=[video_folder_path], start=frame, end=frame + 1,
                                    videotype=source_video_type, view=[view_vertical, view_horizontal], xlim=xlim,
                                    ylim=ylim, zlim=zlim)
        for result_name in result_names:
            result_full_path = os.path.join(video_folder_path, result_name + '.mpg')
            temp_dir = os.path.join(video_folder_path, "temp_" + result_name)
            os.replace(result_full_path, os.path.join(temp_dir, result_name + '_' + str(index) + '.mpg'))

    for result_name in result_names:
        subprocess.call(
            [
                "ffmpeg",
                "-start_number",
                str(start_frame),
                "-framerate",
                str(30),
                "-i",
                os.path.join(video_folder_path, "temp_" + result_name, "img%4d.png"),
                "-c:v",
                "libx264",
                "-vb",
                "20M",
                os.path.join(video_folder_path, str("./result_" + result_name + ".mp4")),
            ]
        )


def apply_kalman_filter(h5_file_path):
    # h5 -> Data Frame
    coordinate_data = read_h5_file(h5_file_path)
    bodyparts, coordinate_names, total_dimension, total_index_number, dataFrame_columns = get_dataframe_indexing_data(
        coordinate_data)
    coordinate_data = get_numpy_data(coordinate_data, total_index_number, len(bodyparts), total_dimension)
    coordinate_results = add_kalman_filter_to_coordinate_data(coordinate_data, total_dimension)
    coordinate_results = np.array(coordinate_results).transpose((1, 0, 2))
    coordinate_results = pack_numpy_data(coordinate_results, total_index_number, len(bodyparts), total_dimension,
                                         dataFrame_columns)
    # Data Frame -> h5
    write_h5_file(coordinate_results, h5_file_path)


def estimate_opposite_eyes(h5_file_path):
    # h5 -> Data Frame
    coordinate_data = read_h5_file(h5_file_path)
    target_bodyparts = ('leftEye', 'rightEye', 'mouth', 'dorsalfin1', 'pelvicfin')
    bodyparts, coordinate_names, total_dimension, total_index_number, dataFrame_columns = get_dataframe_indexing_data(
        coordinate_data)
    target_bodyparts_index = [bodyparts.index(target) for target in target_bodyparts]
    coordinate_data = get_numpy_data(coordinate_data, total_index_number, len(bodyparts), total_dimension)

    eye_result = count_eyes_coordinate_data(coordinate_data, target_bodyparts_index)
    eyes_index = target_bodyparts_index[0:2]
    coordinate_results = coordinate_data.transpose((1, 0, 2))
    eye_result = np.array(eye_result).transpose((1, 0, 2))
    # replace eye data with new data
    for index, eye_index in enumerate(eyes_index):
        coordinate_results[eye_index] = eye_result[index]
    coordinate_results = pack_numpy_data(coordinate_results, total_index_number, len(bodyparts), total_dimension,
                                         dataFrame_columns)
    # Data Frame -> h5
    write_h5_file(coordinate_results, h5_file_path)


def read_h5_file(h5_file_path):
    return pd.read_hdf(h5_file_path)


def write_h5_file(data, file_path, key="df_with_missing"):
    data.to_hdf(file_path, key, format="table", mode="w")


def get_dataframe_indexing_data(dataFrame_data):
    # unpack index/column data
    coordinate_names = tuple(dataFrame_data.columns.get_level_values('coords').unique().tolist())
    return tuple(dataFrame_data.columns.get_level_values('bodyparts').unique().tolist()), coordinate_names, len(
        coordinate_names), len(dataFrame_data.index), dataFrame_data.columns


def get_numpy_data(dataFrame_data, total_index_number, bodyparts_number, total_dimension):
    # Data Frame -> numpy (index, 21*3) -> (index, 21, 3)
    return dataFrame_data.to_numpy().reshape((total_index_number, bodyparts_number, total_dimension)).transpose(
        (1, 0, 2))


def pack_numpy_data(numpy_data, total_index_number, bodyparts_number, total_dimension, dataFrame_columns):
    # numpy (index, 21, 3) -> (index, 21*3)
    packed_numpy_array = numpy_data.reshape(
        (total_index_number, bodyparts_number * total_dimension))
    # numpy -> Data Frame
    return pd.DataFrame(packed_numpy_array, columns=dataFrame_columns)


def count_eyes_coordinate_data(data, target_bodyparts_index):
    result = []
    for index_data in data:
        process_coordinate_data = index_data[target_bodyparts_index, :]
        exists_mask = ~np.all(np.isnan(process_coordinate_data), 1)
        eye_data = process_coordinate_data[0:2]
        eye_mask = exists_mask[0:2]
        if np.all(exists_mask[2:5]) and np.logical_xor(eye_mask[0], eye_mask[1]):
            exists_eye = eye_data[eye_mask]
            unknown_eye = count_opposite_eye_plane_coordinate(process_coordinate_data[2:5], exists_eye)
            eye_data = np.concatenate((exists_eye, unknown_eye), axis=0) if eye_mask[0] else np.concatenate(
                (unknown_eye, exists_eye), axis=0) if unknown_eye.all() else process_coordinate_data[0:2]
            result.append(eye_data.tolist())
        else:
            result.append(process_coordinate_data[0:2].tolist())
    return result


def add_kalman_filter_to_coordinate_data(data, dimension):
    result = []
    for bodyparts_data in data:
        # Kalman filter initialize
        kalman_filter = cv2.KalmanFilter(dimension, dimension, type=cv2.CV_64F)
        kalman_filter.measurementMatrix = np.identity(dimension, np.float64)
        kalman_filter.transitionMatrix = np.identity(dimension, np.float64)
        kalman_filter.processNoiseCov = np.identity(dimension, np.float64) * 1e-3
        kalman_filter.measurementNoiseCov = np.identity(dimension, np.float64) * 1e-2
        kalman_filter.statePre = np.zeros((dimension, 1), dtype=np.float64)

        bodypart_results = []
        start_flag = False
        for index_data in bodyparts_data:
            # NaN process
            if np.isnan(index_data).all():
                bodypart_results.append(
                    kalman_filter.statePost.reshape(dimension, ).tolist() if start_flag else index_data.tolist())
                continue
            else:
                if not start_flag:
                    start_flag = True
                measurement_data = index_data
            # predict and correct circle
            kalman_filter.correct(measurement_data)
            kalman_filter.predict()
            bodypart_results.append(kalman_filter.statePost.reshape(dimension, ).tolist())
        result.append(bodypart_results)
    return result


def count_opposite_eye_plane_coordinate(plane_coordinate, known_eye_coordinate):
    vector_AB = plane_coordinate[1] - plane_coordinate[0]
    vector_AC = plane_coordinate[2] - plane_coordinate[0]
    cross_vector = np.cross(vector_AB, vector_AC)
    if np.linalg.norm(cross_vector) != 0:
        opposite_eye_coordinate = known_eye_coordinate + 2 * (
                np.tensordot(plane_coordinate[0] - known_eye_coordinate, cross_vector, axes=1) /
                np.tensordot(cross_vector, cross_vector, axes=1)) * cross_vector
        return opposite_eye_coordinate
    return np.constant(False)
