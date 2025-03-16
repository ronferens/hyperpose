from dataclasses import dataclass, field
from typing import List, Dict
import os
from PIL import Image
import numpy as np
import plotly.graph_objects as go
from screeninfo import get_monitors

"""
6DoF Camera Pose Visualizer Class
"""


class CameraPoseVisualizer:
    @dataclass
    class Config:
        """
        Configuration class for CameraVisualizer.

        Attributes:
            plot_cam_frustum (bool): Whether to plot the camera frustum.
            plot_cam_axes (bool): Whether to plot the camera axes.
            plot_cam_trace (bool): Whether to plot the camera trace.
            scene_bounds List[float]: The bounds of the scene.
            scale (float): The scale factor for the visualization.
            fov_deg (float): The field of view in degrees.
            disp_frequency (int): The display frequency.
            mesh_z_shift (float): The Z-axis shift for the mesh.
            mesh_scale (float): The scale factor for the mesh.
            show_background (bool): Whether to show the background.
            show_grid (bool): Whether to show the grid.
            show_ticklabels (bool): Whether to show tick labels.
            show_legend (bool): Whether to show the legend.
            mesh_path (object): The path to the mesh file.
            camera_x (float): The X-axis position of the camera.
        """
        # Visualization
        plot_cam_frustum: bool = True
        plot_cam_axes: bool = True
        plot_cam_trace: bool = True

        # Plot properties
        scene_bounds: List[float] = field(default_factory=lambda: [-10.0, 10.0, -10.0, 10.0, -10.0, 10.0])
        scale: float = 1.0
        fov_deg: float = 50.
        disp_frequency: int = 1
        mesh_z_shift: float = 0.0
        mesh_scale: float = 1.0
        show_background: bool = True
        show_grid: bool = True
        show_ticklabels: bool = True
        show_legend: bool = False
        mesh_path: object = None
        camera_x: float = 1.0

    def __init__(self, config: Config = None):
        """
        Initializes the CameraVisualizer with the given configuration.

        :param config: Configuration object for the visualizer.
        """
        self._config = self.Config()
        if config is not None:
            self._config = config

        self._camera_x = self._config.camera_x

        self._traces_poses = None
        self._traces_labels = None
        self._legends = None
        self._traces_color = ["blue", "red", "purple", "orange"]
        self._traces_color_track = ["Blues", "Reds", "Purples", "Oranges"]
        self.MAX_TRACES = len(self._traces_color)

        self._raw_images = None
        self._bit_images = None
        self._image_colorscale = None

        self._camera_set = None

        # Creating a new Plotly figure
        self._fig = None#go.Figure()

        # Mesh setup
        self._mesh = None
        if self._config.mesh_path is not None and os.path.exists(self._config.mesh_path):
            import trimesh
            self._mesh = trimesh.load(self._config.mesh_path, force='mesh')

        if self._mesh is not None:
            self._fig.add_trace(
                go.Mesh3d(
                    x=self._mesh.vertices[:, 0] * self._config.mesh_scale,
                    y=self._mesh.vertices[:, 2] * -self._config.mesh_scale,
                    z=(self._mesh.vertices[:, 1] + self._config.mesh_z_shift) * self._config.mesh_scale,
                    i=self._mesh.faces[:, 0],
                    j=self._mesh.faces[:, 1],
                    k=self._mesh.faces[:, 2],
                    color=None,
                    facecolor=None,
                    opacity=0.8,
                    lighting={'ambient': 1},
                )
            )

    @staticmethod
    def _encode_image(raw_image):
        """
        Encodes a raw image into a format suitable for Plotly.

        :param raw_image: (H, W, 3) array of uint8 in [0, 255].
        :return: Encoded image and colorscale.
        """
        # https://stackoverflow.com/questions/60685749/python-plotly-how-to-add-an-image-to-a-3d-scatter-plot
        dum_img = Image.fromarray(np.ones((3, 3, 3), dtype='uint8')).convert('P', palette='WEB')
        idx_to_color = np.array(dum_img.getpalette()).reshape((-1, 3))

        bit_image = Image.fromarray(raw_image).convert('P', palette='WEB', dither=None)
        # bit_image = Image.fromarray(raw_image.clip(0, 254)).convert(
        #     'P', palette='WEB', dither=None)
        colorscale = [
            [i / 255.0, 'rgb({}, {}, {})'.format(*rgb)] for i, rgb in enumerate(idx_to_color)]

        return bit_image, colorscale

    @staticmethod
    def _project_points_3d(cam_pose: np.array, points: np.array, scale: float = 1.0) -> np.array:
        """
        Projects 3D points into the camera coordinate system.

        :param cam_pose: Camera pose matrix.
        :param points: Array of 3D points.
        :param scale: Scale factor for the points.
        :return: Projected 3D points.
        """
        proj_points = []

        for pnt in points:
            proj_pnt = np.dot(cam_pose[:3, :3], pnt)
            proj_pnt = np.array(proj_pnt) / np.linalg.norm(proj_pnt, ord=2) * scale
            proj_pnt[0] = cam_pose[0, -1] + proj_pnt[0]
            proj_pnt[1] = cam_pose[1, -1] + proj_pnt[1]
            proj_pnt[2] = cam_pose[2, -1] + proj_pnt[2]
            proj_points.append(proj_pnt)

        return proj_points

    def _project_cam_frustum_pts_3d(self, c2w, fov_deg, scale=1.0):
        """
        Projects the camera frustum points into 3D space.

        :param c2w: Camera-to-world transformation matrix.
        :param fov_deg: Field of view in degrees.
        :param scale: Scale factor for the frustum.
        :return: Projected frustum points.
        """
        fov_rad = np.deg2rad(fov_deg)

        corn1 = [np.tan(fov_rad / 2.0), np.tan(fov_rad), 1.0]
        corn2 = [-np.tan(fov_rad / 2.0), np.tan(fov_rad), 1.0]
        corn3 = [-np.tan(fov_rad / 2.0), -np.tan(fov_rad), 1.0]
        corn4 = [np.tan(fov_rad / 2.0), -np.tan(fov_rad), 1.0]
        corn5 = [0, 0, 1.0]

        proj_points = self._project_points_3d(c2w, np.array([corn1, corn2, corn3, corn4, corn5]), scale=scale)
        xs = [c2w[0, -1]]
        ys = [c2w[1, -1]]
        zs = [c2w[2, -1]]
        for pnt in proj_points:
            xs.append(pnt[0])
            ys.append(pnt[1])
            zs.append(pnt[2])

        return np.array([xs, ys, zs]).T

    def _project_cam_orientation_pts_3d(self, c2w, scale=1.0):
        """
        Projects the camera orientation points into 3D space.

        :param c2w: Camera-to-world transformation matrix.
        :param scale: Scale factor for the orientation points.
        :return: Projected orientation points.
        """
        corn1 = [1, 0, 0]
        corn2 = [0, 1, 0]
        corn3 = [0, 0, 1]

        proj_points = self._project_points_3d(c2w, np.array([corn1, corn2, corn3]), scale=scale)
        xs = [c2w[0, -1]]
        ys = [c2w[1, -1]]
        zs = [c2w[2, -1]]
        for pnt in proj_points:
            xs.append(pnt[0])
            ys.append(pnt[1])
            zs.append(pnt[2])

        return np.array([xs, ys, zs]).T

    def update_figure(self,
                      traces_list: List[List[np.array]],
                      traces_labels: List[str],
                      frame_idx: int,
                      camera_pose: Dict = None,
                      images=None):

        # Setting the default camera pose
        fig_camera = dict(eye=dict(x=1.5, y=1.5, z=.2),
                          center=dict(x=0.0, y=0.0, z=0.0),
                          up=dict(x=0.0, y=0.0, z=1.0))
        if camera_pose is not None:
            # Updating camera pose if modified by the user
            fig_camera = camera_pose

        # Initializing the figure
        self._fig = go.Figure()

        if images is not None:
            self._raw_images = images
            self._bit_images = []
            self._image_colorscale = []
            for img in images:
                if img is None:
                    self._bit_images.append(None)
                    self._image_colorscale.append(None)
                    continue

                bit_img, colorscale = self._encode_image(img)
                self._bit_images.append(bit_img)
                self._image_colorscale.append(colorscale)

        # Setting the traces
        self._traces_poses = []
        self._traces_labels = traces_labels
        self._legends = []

        if len(traces_list) > self.MAX_TRACES:
            raise ValueError(f"Number of traces exceeds the maximum number of traces ({self.MAX_TRACES}).")

        for traces_idx, poses in enumerate(traces_list):
            poses = poses[:frame_idx + 1][::self._config.disp_frequency]
            self._traces_poses.append(poses)

            for i in range(len(poses)):
                self._legends.append(f'{traces_labels[traces_idx]}_pose_{i + 1}')

        # ===========================================================
        # Drawing the camera pose - frustum, axes
        # ===========================================================
        if self._config.plot_cam_frustum:
            self._draw_camera_frustum(self._config.fov_deg, self._config.scale)
        if self._config.plot_cam_axes:
            self._draw_camera_axes(self._config.scale)

        # ===========================================================
        # Drawing the camera location and trace
        # ===========================================================
        self._draw_camera_origin(scale=self._config.scale)
        if self._config.plot_cam_trace:
            self._draw_camera_trace()

        # ===========================================================
        # Drawing the corresponding images
        # ===========================================================
        self._fig.update_layout(
            xaxis_autorange="reversed",
            autosize=True,
            hovermode=False,
            margin=go.layout.Margin(l=0, r=0, b=0, t=0),
            showlegend=self._config.show_legend,
            legend=dict(
                yanchor='bottom',
                y=0.01,
                xanchor='right',
                x=0.99,
            ),
            scene=dict(
                aspectmode='manual',
                aspectratio=dict(x=1.5, y=1.5, z=1),
                camera=fig_camera,
                xaxis_title='X-axis',
                yaxis_title='Y-axis',
                zaxis_title='Z-axis',
                xaxis=dict(
                    range=[self._config.scene_bounds[0], self._config.scene_bounds[1]],
                    showticklabels=self._config.show_ticklabels,
                    showgrid=self._config.show_grid,
                    zeroline=False,
                    showbackground=self._config.show_background,
                    showspikes=False,
                    showline=False,
                    ticks=''),
                yaxis=dict(
                    range=[self._config.scene_bounds[2], self._config.scene_bounds[3]],
                    showticklabels=self._config.show_ticklabels,
                    showgrid=self._config.show_grid,
                    zeroline=False,
                    showbackground=self._config.show_background,
                    showspikes=False,
                    showline=False,
                    ticks=''),
                zaxis=dict(
                    # range=[self._config.scene_bounds[4], self._config.scene_bounds[5]],
                    range=[-0.1, self._config.scene_bounds[5]],
                    showticklabels=self._config.show_ticklabels,
                    showgrid=self._config.show_grid,
                    zeroline=False,
                    showbackground=self._config.show_background,
                    showspikes=False,
                    showline=False,
                    ticks='')
            )
        )

    def get_fig(self):
        return self._fig

    def show(self):
        self._fig.show()

    def save_image(self, filename, save_full_screen=True):
        if save_full_screen:
            monitor = get_monitors()
            self._fig.write_image(filename, width=monitor[0].width, height=monitor[0].height, scale=2)
        else:
            self._fig.write_image(filename)

    def _project_camera_frustum(self, pose, fov_deg, scale, draw_p_pnt: bool = False):
        """
        Draws a camera frustum at a given location
        :param pose:
        :param fov_deg:
        :param scale:
        :param draw_p_pnt:
        :return:
        """
        # Projecting the camera frustum according to the given camera pose
        cam_frustum = self._project_cam_frustum_pts_3d(pose, fov_deg, scale=scale)

        # Setting the camera frustum edges' indecies to draw
        edges = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1)]
        if draw_p_pnt:
            edges += [(0, 5)]

        # Drawing the camera frustum according to the calculated camera pose
        cam_frustum_pnts = self._project_camera_pose_visualization(cam_frustum, edges)

        return cam_frustum_pnts

    def _project_camera_axes(self, pose, scale):
        """
        Draws a camera frustum at a given location
        :param pose:
        :param fov_deg:
        :param scale:
        :param draw_p_pnt:
        :return:
        """
        # Projecting the camera 3D axes according to the given camera pose
        cam_axes = self._project_cam_orientation_pts_3d(pose, scale=scale)

        # Setting the camera frustum edges' indecies to draw
        edges = [(0, 1), (0, 2), (0, 3)]

        # Drawing the camera frustum according to the calculated camera pose
        cam_axes_pnts = self._project_camera_pose_visualization(cam_axes, edges)

        return cam_axes_pnts

    @staticmethod
    def _project_camera_pose_visualization(points_3d: np.array, edges: List):
        """
        Draws a camera frustum at a given location
        :param points_3d:
        :param edges:
        :return:
        """
        # Drawing the camera frustum according to the calculated camera pose
        projected_points_3d = []
        for (idx, edge) in enumerate(edges):
            (x1, x2) = (points_3d[edge[0], 0], points_3d[edge[1], 0])
            (y1, y2) = (points_3d[edge[0], 1], points_3d[edge[1], 1])
            (z1, z2) = (points_3d[edge[0], 2], points_3d[edge[1], 2])
            projected_points_3d.append({'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'z1': z1, 'z2': z2})

        return projected_points_3d

    def _draw_camera_frustum(self, fov_deg, scale):
        for trace_idx, poses in enumerate(self._traces_poses):
            cam_poses = []

            for i in range(len(poses)):
                pose = poses[i]

                # Calculate the projected camera frustum based on the given 6DoF
                cam_frustum_pnts = self._project_camera_frustum(pose, fov_deg, scale)

                # Adding the projected points4
                cam_poses.append(cam_frustum_pnts)


            # Prepare the x, y, and z coordinates for the lines
            x, y, z = [], [], []
            for cam_frustum in cam_poses:
                for pnt in cam_frustum:
                    x.extend([pnt['x1'], pnt['x2'], None])  # Add origin, destination, None
                    y.extend([pnt['y1'], pnt['y2'], None])
                    z.extend([pnt['z1'], pnt['z2'], None])

            self._fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z, mode='lines',
                line=dict(color='black', width=3),
                name=f'Camera Frustum - {self._traces_labels[trace_idx]}', showlegend=True))

    def _draw_camera_axes(self, scale):
        for trace_idx, poses in enumerate(self._traces_poses):
            cam_poses = []

            for i in range(len(poses)):
                pose = poses[i]

                # Calculate the projected camera frustum based on the given 6DoF
                cam_axes_pnts = self._project_camera_axes(pose, scale)

                # Adding the projected points4
                cam_poses.append(cam_axes_pnts)

        # Function to calculate arrowhead points
        def _get_arrowhead(start, end, scale=0.1):
            """
            Calculate coordinates for an arrowhead.
            :param start: Starting point (x, y, z) of the arrow.
            :param end: Ending point (x, y, z) of the arrow.
            :param scale: Scale factor for the size of the arrowhead.
            :return: Coordinates for the arrowhead points.
            """
            start = np.array(start)
            end = np.array(end)
            direction = end - start
            direction = direction / np.linalg.norm(direction)  # Normalize the direction vector

            # Perpendicular vector for the arrowhead base
            perp = np.cross(direction, [1, 0, 0])
            if np.linalg.norm(perp) < 1e-5:  # Handle edge case where direction is parallel to [1, 0, 0]
                perp = np.cross(direction, [0, 1, 0])
            perp = perp / np.linalg.norm(perp) * scale

            # Calculate arrowhead base points
            arrow_base = end - direction * scale
            p1 = arrow_base + perp
            p2 = arrow_base - perp

            return [end, p1, end, p2]  # Arrow tip to base points

        # Prepare the x, y, and z coordinates for the lines
        x, y, z = [], [], []
        for idx, cam_frustum in enumerate(cam_poses):
            for pnt in cam_frustum:
                x.extend([pnt['x1'], pnt['x2'], None])  # Add origin, destination, None
                y.extend([pnt['y1'], pnt['y2'], None])
                z.extend([pnt['z1'], pnt['z2'], None])
                start, end = [pnt['x1'], pnt['y1'], pnt['z1']], [pnt['x2'], pnt['y2'], pnt['z2']]

                # Add arrowhead
                arrowhead = _get_arrowhead(start, end, scale=0.05)
                for i in range(0, len(arrowhead), 2):
                    x.extend([arrowhead[i][0], arrowhead[i + 1][0], None])
                    y.extend([arrowhead[i][1], arrowhead[i + 1][1], None])
                    z.extend([arrowhead[i][2], arrowhead[i + 1][2], None])

        self._fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z, mode='lines',
            line=dict(color='black', width=3),
            name=f'Camera Axes - {self._traces_labels[trace_idx]}', showlegend=True))

    def _draw_camera_origin(self, scale: int = 20):
        for trace_idx, poses in enumerate(self._traces_poses):
            x, y, z = [], [], []
            for pose in poses:
                x.append(pose[0, -1])
                y.append(pose[1, -1])
                z.append(pose[2, -1])

            color = 'green' if self._traces_labels[trace_idx].lower() == 'gt' else self._traces_color[trace_idx]
            self._fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z, mode='markers',#+text',  # Add both markers and text
                marker=dict(
                    size=3,
                    color=color,  # Use z values for color
                    opacity=0.8,
                ),
                text=self._legends,  # Assign text labels
                textfont=dict(
                    size=scale,  # Size of the text
                    color='black'  # Color of the text
                ),
                textposition="top center",  # Position text above the markers
                name=f'{self._traces_labels[trace_idx].title()}', showlegend=True))

    def _draw_camera_trace(self):
        for trace_idx, poses in enumerate(self._traces_poses):
            # Prepare the x, y, and z coordinates for the lines
            x_line, y_line, z_line = [], [], []
            num_points = len(poses)
            intensity = np.linspace(0.25, 0.75, 3 * num_points)

            for idx in range(num_points - 1):
                x_line.extend([poses[idx][0, -1], poses[idx + 1][0, -1], None])  # Add origin, destination, None
                y_line.extend([poses[idx][1, -1], poses[idx + 1][1, -1], None])
                z_line.extend([poses[idx][2, -1], poses[idx + 1][2, -1], None])

            color = 'Greens' if self._traces_labels[trace_idx].lower() == 'gt' else self._traces_color_track[trace_idx]
            self._fig.add_trace(go.Scatter3d(
                x=x_line, y=y_line, z=z_line, mode='lines',
                line=dict(
                    color=intensity,  # Use intensity for color mapping
                    colorscale=color,  # Color scale for the line
                    width=5
                ),
                name=f'Camera Traces - {self._traces_labels[trace_idx]}', showlegend=False))
