import argparse
import json
import dash
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
import cv2
import base64
from util.cam_pose_visualizer import CameraPoseVisualizer
from util.utils import quaternion_to_rotation_matrix
import numpy as np
from dataclasses import dataclass
from typing import List
import pandas as pd

"""
Image Sequence Player
"""


class ImageSequencePlayer:
    """
    Image Sequence Player Class
    This class handles the loading and display of image sequences.
    """

    def __init__(self, _image_files):
        """
        Constructor
        :param _image_files: A list of images to display
        """
        # Get sorted list of image files
        self._image_files = _image_files
        self._total_frames = len(self._image_files)

    def get_total_frames(self):
        """
        Returns the total number of frames in the sequence
        :return: Total number of frames
        """
        return self._total_frames

    def get_frame(self, frame_idx):
        """
        Returns the frame at the given index
        :param frame_idx: Index of the next frame to load
        :return: Base64 encoded image
        """
        if 0 <= frame_idx < self._total_frames:
            img_path = self._image_files[frame_idx]

            # Encode frame as Base64 for Dash
            _, buffer = cv2.imencode(".jpg", cv2.imread(img_path))
            encoded_image = base64.b64encode(buffer).decode("utf-8")

            return f'data:image/jpeg;base64,{encoded_image}'
        return None


"""
Camera Pose Visualization App
"""


class CamPoseVizApp:
    """
    Camera Pose Visualization App Class
    This class handles the visualization of camera poses using a Dash application.
    """

    @dataclass
    class Data:
        """
        Data class to store input data
        """
        traces: List[List[np.ndarray]]
        labels: List[List[np.ndarray]]
        timestamps: List[int]

    def __init__(self, input: str, data: Data, config: CameraPoseVisualizer.Config, scene_name: str):
        """
        Constructor
        :param input: Input data to process
        :param data: Pose data to visualize
        :param config: Camera visualizer configuration
        """
        self._camera_viz = CameraPoseVisualizer(config=config)
        self._player = ImageSequencePlayer(input)
        self._scene_name = scene_name

        # Saving the input data
        self._data = data

        # Create Dash app
        self._app = Dash(__name__)
        self._set_app_layout()

        self._app.callback(
            Output('camera-pose', 'data'),
            Input('scatter-3d', 'relayoutData'),
            prevent_initial_call=True)(
            self._callback_store_camera_position)

        self._app.callback(
            [Output('frame-display', 'src'),
             Output('frame-slider', 'value'),
             Output('scatter-3d', 'figure'),
             Output('frame-info', 'children')],
            [Input('current-frame', 'data'),
             Input('camera-pose', 'data')])(
            self._callback_update_display)

        self._app.callback(Output('playback-interval', 'disabled'),
                           Input('play-button', 'n_clicks'),
                           Input('pause-button', 'n_clicks'))(
            self._callback_toggle_playback)

        self._app.callback(Output('current-frame', 'data'),
                           Input('playback-interval', 'n_intervals'),
                           Input('frame-slider', 'value'),
                           State('current-frame', 'data'))(
            self._callback_update_frame)

    def _set_app_layout(self):
        """
        Sets the layout of the Dash app
        :return: None
        """
        self._app.layout = html.Div([
            html.H1(self._scene_name),
            # Playback controls
            html.Div([
                html.Button('Play', id='play-button', n_clicks=0),
                html.Button('Pause', id='pause-button', n_clicks=0),
                dcc.Slider(
                    id='frame-slider',
                    min=0,
                    max=self._player.get_total_frames() - 1,
                    value=0,
                    step=1,
                    marks={i: str(i) for i in range(0, self._player.get_total_frames(), 100)}
                ),
            ], style={'margin': '20px 0'}),

            html.Div([
                html.Img(id='frame-display', style={'width': '40%', 'border': '2px solid black'}),
                dcc.Graph(id='scatter-3d', config={'scrollZoom': True},
                          style={'width': '60%', 'border': '2px solid black'})
            ], style={'display': 'flex', 'justify-content': 'space-between', 'align-items': 'center-top',
                      'width': '100%'}),

            # Current frame display
            html.Div(id="frame-info", style={'margin': '10px 0'}),

            dcc.Interval(id='playback-interval', interval=100, disabled=True),  # Updates every 100ms

            # Store current frame
            dcc.Store(id='current-frame', data=0),

            # Store camera pose
            dcc.Store(id='camera-pose')
        ])

    def run(self):
        """
        Runs the Dash app
        :return: None
        """
        self._app.run_server(debug=True)

    # ============================
    # CALLBACKS
    # ============================
    def _callback_store_camera_position(self, relayout_data):
        if relayout_data and "scene.camera" in relayout_data:
            return json.dumps(relayout_data["scene.camera"])
        return dash.no_update

    def _callback_update_display(self, frame_idx, camera_pose):
        """
        Updates the display based on the current frame index
        :param frame_idx: The current frame index
        :param camera_pose: The current camera pose to set in the 3D scatter plot
        :return: Base64 encoded image, updated frame index, updated graph, and frame info
        """
        # Update image
        img_src = self._player.get_frame(frame_idx)

        # Maintain the current camera pose (if modified by the user)
        current_camera_pose = json.loads(camera_pose) if camera_pose else None

        # Update graph
        self._camera_viz.update_figure(traces_list=self._data.traces,
                                       traces_labels=self._data.labels,
                                       frame_idx=frame_idx,
                                       camera_pose=current_camera_pose)
        fig = self._camera_viz.get_fig()

        frame_info = f"Current Frame Index: {frame_idx}/{self._player.get_total_frames() - 1}"

        return img_src, frame_idx, fig, frame_info

    @staticmethod
    def _callback_toggle_playback(play_clicks, pause_clicks):
        """
        Toggles the playback state based on the play/pause button clicks
        :param play_clicks: The number of times the play button was clicked
        :param pause_clicks: The number of times the pause button was clicked
        :return: Boolean indicating whether the playback interval is disabled
        """
        if dash.callback_context.triggered_id == 'play-button':
            return False
        return True

    def _callback_update_frame(self, n_intervals, slider_value, current_frame):
        """
        Updates the current frame based on the playback interval
        :param n_intervals: Number of intervals
        :param slider_value: Slider current value
        :param current_frame: Current frame index
        :return: Index of the next frame to display
        """
        # If slider was moved, use its value
        if dash.callback_context.triggered_id == 'frame-slider':
            return slider_value
        # Otherwise increment frame, but do not loop back to the start
        next_frame = current_frame + 1
        if next_frame >= self._player.get_total_frames():
            return current_frame  # Stop at the last frame
        return next_frame


##############################
# Main App
##############################
def parse_pose_data(filename, is_world2cam=False):
    """
    Parses a file with lines formatted as:
    img_filename, x, y, z, qx, qy, qz, qw

    :param filename: Path to the text file
    :param is_world2cam: Flag to indicate if the poses are in World2Cam format
    :return: List of dictionaries containing parsed data
    """
    est_poses = []
    gt_poses = []
    img_filenames = []

    max_x = None
    max_y = None
    max_z = None
    min_x = None
    min_y = None
    min_z = None

    df = pd.read_csv(filename)
    if df is not None:
        for index, row in df.iterrows():
            # Retrieving estimated pose data
            pose = np.zeros((4, 4))
            pose[-1, -1] = 1
            rot_mat = quaternion_to_rotation_matrix(qx=row["est_qw"], qy=row["est_qy"], qz=row["est_qz"],
                                                    qw=row["est_qw"])
            pose[:3, :3] = rot_mat
            pose[:3, -1] = np.array([row["est_x"], row["est_y"], row["est_z"]])

            # Convert World2Cam to Cam2World
            if is_world2cam:
                pose[:3, :3] = -1 * np.dot(np.linalg.inv(rot_mat), pose[:3, -1])
                pose[:3, -1] = -1 * pose[:3, -1]

            est_poses.append(pose)
            img_filenames.append(row["img_filename"])

            # Retrieving ground-truth pose data
            pose = np.zeros((4, 4))
            pose[-1, -1] = 1
            rot_mat = quaternion_to_rotation_matrix(qx=row["gt_qw"], qy=row["gt_qy"], qz=row["gt_qz"],
                                                    qw=row["gt_qw"])
            pose[:3, :3] = rot_mat
            pose[:3, -1] = np.array([row["gt_x"], row["gt_y"], row["gt_z"]])

            # Convert World2Cam to Cam2World
            if is_world2cam:
                pose[:3, :3] = -1 * np.dot(np.linalg.inv(rot_mat), pose[:3, -1])
                pose[:3, -1] = -1 * pose[:3, -1]

            gt_poses.append(pose)

            if max_x is None or max_x < pose[0, -1]:
                max_x = pose[0, -1]
            if max_y is None or max_y < pose[1, -1]:
                max_y = pose[1, -1]
            if max_z is None or max_z < pose[2, -1]:
                max_z = pose[2, -1]
            if min_x is None or min_x > pose[0, -1]:
                min_x = pose[0, -1]
            if min_y is None or min_y > pose[1, -1]:
                min_y = pose[1, -1]
            if min_z is None or min_z > pose[2, -1]:
                min_z = pose[2, -1]
    else:
        print("No DataFrame to parse.")
        return None, None

    # Scaling the scene bounds
    min_x = 0.9 * min_x if min_x > 0 else 1.1 * min_x
    min_y = 0.9 * min_y if min_y > 0 else 1.1 * min_y
    min_z = 0.9 * min_z if min_z > 0 else 1.1 * min_z
    max_x = 0.9 * max_x if max_x < 0 else 1.1 * max_x
    max_y = 0.9 * max_y if max_y < 0 else 1.1 * max_y
    max_z = 0.9 * max_z if max_z < 0 else 1.1 * max_z

    # Setting the scene bounds based on the max and min values
    scene_bounds = [min_x, max_x, min_y, max_y, min_z, max_z]

    return est_poses, gt_poses, img_filenames, scene_bounds


def run_camera_pose_viz_app():
    """
    Runs the Camera Pose Visualization App
    :return: None
    """
    poses_list = []
    poses_labels = []

    for trace_idx, poses_file in enumerate(args.poses_files):
        # Retrieves the camera pose to visualize
        poses, gt_poses, img_filenames, scene_bounds = parse_pose_data(poses_file)
        poses_list.append(poses)
        poses_labels.append(args.poses_labels[trace_idx])
    poses_list.append(gt_poses)
    poses_labels.append('gt')

    # Creating the camera pose visualization app
    cam_pose_app = CamPoseVizApp(input=img_filenames,
                                 data=CamPoseVizApp.Data(traces=poses_list,
                                                         labels=poses_labels,
                                                         timestamps=np.arange(len(poses))),
                                 config=CameraPoseVisualizer.Config(plot_cam_axes=False,
                                                                    plot_cam_frustum=False,
                                                                    show_legend=True,
                                                                    scale=1,
                                                                    disp_frequency=1,
                                                                    scene_bounds=scene_bounds),
                                 scene_name=f'Camera Pose Visualization : {args.scene_name}')

    # Running the camera pose visualization app
    cam_pose_app.run()


if __name__ == '__main__':
    """
    This script runs the Camera Pose Visualization App.
    The app displays the camera poses and the corresponding video frames.
    """
    app_Args = argparse.ArgumentParser()
    app_Args.add_argument('-p', '--poses_files', type=str, nargs="+", required=True, help='Input poses file to process')
    app_Args.add_argument('-l', '--poses_labels', type=str, nargs="+", required=True,
                          help='Input poses file to process')
    app_Args.add_argument('-n', '--scene_name', type=str, required=True, help='Name of the scene')
    args = app_Args.parse_args()

    run_camera_pose_viz_app()
