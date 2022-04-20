"""
Based on https://github.com/seoungwugoh/ivs-demo

The entry point for the user interface
It is terribly long... GUI code is hard to write!
"""

import cv2
import functools
import json
import numpy as np
import os
import time
import torch
import sys

from argparse import ArgumentParser
from collections import deque
from enum import Enum

from PyQt5.QtWidgets import (
    QWidget,
    QApplication,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QPlainTextEdit,
    QVBoxLayout,
    QSizePolicy,
    QShortcut,
    QProgressBar,
)

from PyQt5.QtGui import QPixmap, QKeySequence, QImage, QTextCursor
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal

from qt_material import apply_stylesheet

from inference_core import InferenceCore
from interact.s2m_controller import S2MController
from interact.fbrs_controller import FBRSController
from model.propagation.prop_net import PropagationNetwork
from model.fusion_net import FusionNet
from model.s2m.s2m_network import deeplabv3plus_mobilenet as S2M
from util.palette import pal_color_map

from masks_manipulation.relevant_points import extract_centers
from masks_manipulation import movement_index

from interact.interactive_utils import *
from interact.interaction import *

import matplotlib

matplotlib.use("Qt5Agg")

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

torch.set_grad_enabled(False)
matplotlib.pyplot.style.use("ggplot")

# DAVIS palette
palette = pal_color_map()


class State(Enum):
    INITIAL = 0
    RECORDED = 1
    PROPAGATED = 2
    FINAL = 3


class FingerMovementsCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=300):
        self.fig = matplotlib.figure.Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.subplots(
            nrows=1, ncols=2, gridspec_kw={"width_ratios": [3, 1]}
        )
        self.fig.subplots_adjust(
            left=0.05, bottom=0.05, right=0.95, top=0.85, wspace=0.1, hspace=0.1
        )
        self.axes[0].tick_params(
            labelsize=3, direction="in", pad=0.2, width=0.5, length=2
        )
        self.axes[0].grid(True, linewidth=0.5, color="#FFFFFF")
        self.axes[1].tick_params(
            labelsize=3, direction="in", pad=0.2, width=0.5, length=2
        )
        self.axes[1].grid(True, linewidth=0.5, color="#FFFFFF")
        self.axes[0].set_axis_off()
        self.axes[1].set_axis_off()
        super().__init__(self.fig)

    def fill(self, finger_centers, labels):
        for i, finger in enumerate(finger_centers):
            self.axes[0].plot(-finger[:, 0], linewidth=0.5)
            self.axes[1].plot(finger[:, 1], range(len(finger)), linewidth=0.5)

        vert_max = np.max(np.abs(finger_centers[:, :, 0]))
        horz_max = np.max(np.abs(finger_centers[:, :, 1]))

        if vert_max < 20:
            self.axes[0].set_ylim([-20, 20])
        if horz_max < 10:
            self.axes[1].set_xlim([-20, 20])

        self.axes[0].set_axis_on()
        self.axes[1].set_axis_on()

        self.axes[0].set_title("Vertical", fontsize=4, loc="center", y=0.93)

        self.axes[1].set_ylim([0, len(finger)])
        self.axes[1].invert_yaxis()
        self.axes[1].set_title("Horizontal", fontsize=4, loc="center", y=0.93)

        self.axes[0].legend(labels=labels, fontsize=3)

    def clear(self):
        self.axes[0].cla()
        self.axes[1].cla()
        self.axes[0].set_axis_off()
        self.axes[1].set_axis_off()

    def set_title(self, title):
        self.fig.suptitle(title, fontsize=5, y=0.96)


class HeatmapCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=300):
        self.fig = matplotlib.figure.Figure(figsize=(width, height), dpi=dpi)
        self.ax = matplotlib.axes.Axes(self.fig, [0.1, 0.05, 0.8, 0.8])
        self.fig.add_axes(self.ax)
        super().__init__(self.fig)
        self.ax.set_axis_off()

    def fill(self, masks, first_frame):
        masks_nonzeros = (masks != 0).astype(int)
        heatmap = masks_nonzeros.sum(axis=0)
        self.ax.imshow(first_frame)
        self.ax.matshow(heatmap, alpha=0.8)

    def clear(self):
        self.ax.cla()
        self.ax.set_axis_off()

    def set_title(self, title):
        self.fig.suptitle(title, fontsize=5, y=0.93)


class VideoCapturer(QThread):
    changePixmap = pyqtSignal((np.ndarray, int))
    disableGUI = pyqtSignal(bool)

    def __init__(self, parent=None, n_frames=300, delay=1):
        super().__init__()
        self.n_frames = n_frames
        self.delay = delay
        self.parent = parent

    def run(self):
        self.disableGUI.emit(True)
        self.cap = cv2.VideoCapture(0)

        start_time = time.time()
        curr_time = time.time()

        curr_diff = curr_time - start_time

        while curr_diff < self.delay:
            ret, self.frame = self.cap.read()
            if ret:
                self.frame = cv2.flip(self.frame, 1)
                rgbimage = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                rgbimage = cv2.resize(rgbimage, (711, 400))
                # Draw a big number in the center for the first frames
                font = cv2.FONT_HERSHEY_DUPLEX
                text = str(self.delay - np.floor(curr_diff).astype(int))
                # get boundary of this text
                textsize = cv2.getTextSize(text, font, 6, 15)[0]
                # get coords based on boundary
                textX = (rgbimage.shape[1] - textsize[0]) // 2
                textY = (rgbimage.shape[0] + textsize[1]) // 2
                # add text centered on image
                cv2.putText(
                    rgbimage, text, (textX, textY), font, 6, (255, 255, 255), 15
                )
                self.changePixmap.emit(rgbimage, -1)
                curr_time = time.time()
                curr_diff = curr_time - start_time

        self.parent.main_canvas.setStyleSheet("border: 2px solid red;")
        for i in range(self.n_frames):
            ret, self.frame = self.cap.read()
            if ret:
                self.frame = cv2.flip(self.frame, 1)
                rgbimage = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                rgbimage = cv2.resize(rgbimage, (711, 400))
                self.changePixmap.emit(rgbimage, i)

        self.parent.main_canvas.setStyleSheet("border: 0px;")
        self.cap.release()
        self.disableGUI.emit(False)


class App(QWidget):
    def __init__(
        self,
        prop_net,
        fuse_net,
        s2m_ctrl: S2MController,
        fbrs_ctrl: FBRSController,
        starting_image,
        vid_len,
        num_objects,
        mem_freq,
        mem_profile,
    ):

        super().__init__()

        self.starting_image = starting_image
        self.num_frames = vid_len
        self.images = np.empty((self.num_frames, 400, 711, 3))
        self.num_objects = num_objects
        self.prop_net = prop_net
        self.fuse_net = fuse_net
        self.s2m_controller = s2m_ctrl
        self.fbrs_controller = fbrs_ctrl
        self.mem_freq = mem_freq
        self.mem_profile = mem_profile

        self.height, self.width = self.starting_image.shape[:2]

        # set window
        self.setWindowTitle("MiVOS")
        self.setGeometry(100, 100, self.width, self.height + 100)

        # Corporative logo
        logo = QLabel()
        logo.setPixmap(QPixmap(os.path.join("assets", "logo_dasci.png")))

        # Language selection buttons
        self.spanish_button = QPushButton("Español")
        self.spanish_button.clicked.connect(lambda: self.select_language("spanish"))
        self.english_button = QPushButton("English")
        self.english_button.clicked.connect(lambda: self.select_language("english"))

        # Languages selector
        languages = QHBoxLayout()
        languages.addWidget(self.spanish_button)
        languages.addWidget(self.english_button)
        languages.addStretch(1)

        # Recording button
        self.record_button = QPushButton()
        self.record_button.clicked.connect(self.on_record)

        # Interaction with video buttons
        self.play_button = QPushButton()
        self.play_button.clicked.connect(self.on_play)
        self.run_button = QPushButton()
        self.run_button.clicked.connect(self.on_run)

        self.compute_button = QPushButton()
        self.compute_button.clicked.connect(self.on_compute)

        self.reset_button = QPushButton()
        self.reset_button.setProperty("class", "danger")
        self.reset_button.clicked.connect(self.on_reset)

        self.undo_button = QPushButton()
        self.undo_button.clicked.connect(self.on_undo)

        # LCD
        self.lcd = QTextEdit()
        self.lcd.setReadOnly(True)
        self.lcd.setMaximumHeight(45)
        self.lcd.setMinimumHeight(45)
        self.lcd.setMaximumWidth(110)
        self.lcd.setMinimumWidth(110)
        self.lcd.setText("{: 3d} / {: 3d}".format(0, self.num_frames - 1))

        # brush size
        self.brush_size = 3

        # Main canvas -> QLabel
        self.main_canvas = QLabel()
        self.main_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.main_canvas.setAlignment(Qt.AlignCenter)
        self.main_canvas.setMinimumSize(100, 100)

        self.main_canvas.mousePressEvent = self.on_press
        self.main_canvas.mouseMoveEvent = self.on_motion
        self.main_canvas.mouseReleaseEvent = self.on_release
        self.main_canvas.setMouseTracking(False)  # Required for all-time tracking

        # Console on the GUI
        self.console = QPlainTextEdit()
        self.console.setReadOnly(True)
        self.console.setMinimumHeight(100)
        self.console.setMaximumHeight(100)

        # Graph and heatmap canvases
        self.finger_movements_canvas = FingerMovementsCanvas(self)
        self.heatmap_canvas = HeatmapCanvas(self)

        # progress bar
        self.progress = QProgressBar(self)
        self.progress.setGeometry(0, 0, 300, 25)
        self.progress.setMinimumWidth(300)
        self.progress.setMinimum(0)
        self.progress.setMaximum(100)
        self.progress.setFormat("Idle")
        self.progress.setStyleSheet("QProgressBar{color: black;}")
        self.progress.setAlignment(Qt.AlignCenter)

        # navigator
        navi = QHBoxLayout()
        navi.addWidget(self.lcd)
        navi.addWidget(self.record_button)
        navi.addWidget(self.play_button)
        navi.addStretch(1)
        navi.addWidget(self.run_button)
        navi.addWidget(self.undo_button)
        navi.addStretch(1)
        navi.addWidget(self.progress)
        navi.addStretch(1)
        navi.addWidget(self.compute_button)
        navi.addWidget(self.reset_button)

        # Language selectors, drawing area and navigation bar
        left_column = QVBoxLayout()
        left_column.addWidget(logo)
        left_column.addLayout(languages)
        left_column.addWidget(self.main_canvas)
        left_column.addLayout(navi)

        # Right bar
        right_column = QVBoxLayout()
        right_column.setAlignment(Qt.AlignTop)
        right_column.addWidget(self.finger_movements_canvas)
        right_column.addWidget(self.heatmap_canvas)
        right_column.addWidget(self.console)

        # Application layout
        layout = QHBoxLayout()
        layout.addLayout(left_column, 3)
        layout.addLayout(right_column, 2)
        self.setLayout(layout)

        # timer
        self.timer = QTimer()
        self.timer.setSingleShot(False)
        self.timer.timeout.connect(self.on_time)

        # Initialize class variables
        self.state = State(0)
        self.refresh_enabled_buttons()
        self.reset_initial_state()

        # Objects shortcuts
        for i in range(1, num_objects + 1):
            QShortcut(QKeySequence(str(i)), self).activated.connect(
                functools.partial(self.hit_number_key, i)
            )

        # <- and -> shortcuts
        QShortcut(QKeySequence(Qt.Key_Left), self).activated.connect(self.on_prev)
        QShortcut(QKeySequence(Qt.Key_Right), self).activated.connect(self.on_next)

        self.interacted_mask = None
        self.waiting_to_start = True

        self.show_starting_image()
        self.select_language("spanish")
        self.show()

    def resizeEvent(self, event):
        self.show_starting_image()

    def select_language(self, language):
        with open(os.path.join("assets", f"{language}_texts.json"), "r") as f:
            self.texts = json.load(f)

        self.refresh_ui_labels()

    def refresh_ui_labels(self):
        self.record_button.setText(self.texts["record_button_label"])
        self.play_button.setText(self.texts["play_button_play_label"])
        self.run_button.setText(self.texts["run_button_label"])
        self.undo_button.setText(self.texts["undo_button_label"])
        self.compute_button.setText(self.texts["compute_button_label"])
        self.reset_button.setText(self.texts["reset_button_label"])
        self.console.clear()
        self.console_push_text(self.texts["console_language_info"])
        self.console_push_text(self.texts["console_init_text"])
        self.finger_movements_canvas.set_title(self.texts["movement_canvas_title"])
        self.finger_movements_canvas.draw()
        self.heatmap_canvas.set_title(self.texts["heatmap_canvas_title"])
        self.heatmap_canvas.draw()

    def show_starting_image(self):
        height, width, channel = self.starting_image.shape
        bytesPerLine = channel * width
        qImg = QImage(
            self.starting_image.data, width, height, bytesPerLine, QImage.Format_RGB888
        )

        self.main_canvas.setPixmap(
            QPixmap(
                qImg.scaled(
                    self.main_canvas.size(), Qt.KeepAspectRatio, Qt.FastTransformation
                )
            )
        )

        self.main_canvas_size = self.main_canvas.size()
        self.image_size = qImg.size()

    def reset_initial_state(self):
        self.current_mask = np.zeros(
            (self.num_frames, self.height, self.width), dtype=np.uint8
        )
        self.vis_map = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.vis_alpha = np.zeros((self.height, self.width, 1), dtype=np.float32)
        self.brush_vis_map = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.brush_vis_alpha = np.zeros((self.height, self.width, 1), dtype=np.float32)
        self.vis_hist = deque(maxlen=100)
        self.cursur = 0
        self.on_showing = None

        # initialize action
        self.interactions = {}
        self.interactions["interact"] = [[] for _ in range(self.num_frames)]
        self.interactions["annotated_frame"] = []
        self.this_frame_interactions = []
        self.interaction = None
        self.reset_this_interaction()
        self.pressed = False
        self.right_click = False
        self.ctrl_size = False
        self.current_object = 1
        self.last_ex = self.last_ey = 0

    def on_record(self):
        self.finger_movements_canvas.clear()
        self.finger_movements_canvas.draw()
        self.heatmap_canvas.clear()
        self.heatmap_canvas.draw()
        self.recorder = VideoCapturer(self, self.num_frames)
        self.recorder.changePixmap.connect(self.set_image)
        self.recorder.disableGUI.connect(self.disable_gui_for_record)
        self.recorder.start()
        self.state = State.RECORDED
        self.refresh_enabled_buttons()

    def disable_gui_for_record(self, disable_gui):
        self.play_button.setEnabled(not disable_gui)
        self.record_button.setEnabled(not disable_gui)
        self.run_button.setEnabled(not disable_gui)
        self.reset_button.setEnabled(not disable_gui)
        self.main_canvas.setMouseTracking(not disable_gui)
        if not disable_gui:
            self.reset_initial_state()
            self.clear_visualization()
            self.show_current_frame()
            self.processor = InferenceCore(
                self.prop_net,
                self.fuse_net,
                images_to_torch(self.images, device="cpu"),
                self.num_objects,
                mem_freq=self.mem_freq,
                mem_profile=self.mem_profile,
            )

    def set_image(self, image, frame_id):
        self.viz = image
        if frame_id >= 0:
            self.images[frame_id] = image

        self.update_interact_vis()

    def console_push_text(self, text):
        self.console.appendPlainText(text)
        self.console.moveCursor(QTextCursor.End)
        print(text)

    def compose_current_im(self):
        self.viz = overlay_davis(
            self.images[self.cursur], self.current_mask[self.cursur]
        )

    def update_interact_vis(self):
        # Update the interactions without re-computing the overlay
        height, width, channel = self.viz.shape
        bytesPerLine = 3 * width

        vis_map = self.vis_map
        vis_alpha = self.vis_alpha
        brush_vis_map = self.brush_vis_map
        brush_vis_alpha = self.brush_vis_alpha

        self.viz_with_stroke = self.viz * (1 - vis_alpha) + vis_map * vis_alpha
        self.viz_with_stroke = (
            self.viz_with_stroke * (1 - brush_vis_alpha)
            + brush_vis_map * brush_vis_alpha
        )
        self.viz_with_stroke = self.viz_with_stroke.astype(np.uint8)

        qImg = QImage(
            self.viz_with_stroke.data, width, height, bytesPerLine, QImage.Format_RGB888
        )
        self.main_canvas.setPixmap(
            QPixmap(
                qImg.scaled(
                    self.main_canvas.size(), Qt.KeepAspectRatio, Qt.FastTransformation
                )
            )
        )

        self.main_canvas_size = self.main_canvas.size()
        self.image_size = qImg.size()

    def show_current_frame(self):
        # Re-compute overlay and show the image
        self.compose_current_im()
        self.update_interact_vis()
        self.lcd.setText("{: 3d} / {: 3d}".format(self.cursur, self.num_frames - 1))

    def get_scaled_pos(self, x, y):
        # Un-scale and un-pad the label coordinates into image coordinates
        oh, ow = self.image_size.height(), self.image_size.width()
        nh, nw = self.main_canvas_size.height(), self.main_canvas_size.width()

        h_ratio = nh / oh
        w_ratio = nw / ow
        dominate_ratio = min(h_ratio, w_ratio)

        # Solve scale
        x /= dominate_ratio
        y /= dominate_ratio

        # Solve padding
        fh, fw = nh / dominate_ratio, nw / dominate_ratio
        x -= (fw - ow) / 2
        y -= (fh - oh) / 2

        x = max(0, min(self.width - 1, x))
        y = max(0, min(self.height - 1, y))

        # return int(round(x)), int(round(y))
        return x, y

    def clear_visualization(self):
        self.vis_map.fill(0)
        self.vis_alpha.fill(0)
        self.vis_hist.clear()
        self.vis_hist.append((self.vis_map.copy(), self.vis_alpha.copy()))

    def reset_this_interaction(self):
        self.complete_interaction()
        self.clear_visualization()
        self.interaction = None
        self.this_frame_interactions = []
        self.undo_button.setDisabled(True)
        if self.fbrs_controller is not None:
            self.fbrs_controller.unanchor()

    def progress_step_cb(self):
        self.progress_num += 1
        ratio = self.progress_num / self.progress_max
        self.progress.setValue(int(ratio * 100))
        self.progress.setFormat("%2.1f%%" % (ratio * 100))
        QApplication.processEvents()

    def progress_total_cb(self, total):
        self.progress_max = total
        self.progress_num = -1
        self.progress_step_cb()

    def on_run(self):
        if self.interacted_mask is None:
            self.console_push_text(self.texts["console_propagate_error"])
            return

        self.console_push_text(self.texts["console_propagate_start"])
        self.set_navi_enable(False)
        self.current_mask = self.processor.interact(
            self.interacted_mask,
            self.cursur,
            self.progress_total_cb,
            self.progress_step_cb,
        )
        self.interacted_mask = None
        # clear scribble and reset
        self.show_current_frame()
        self.reset_this_interaction()
        self.progress.setFormat("Idle")
        self.progress.setValue(0)
        self.console_push_text(self.texts["console_propagate_end"])
        self.state = State.PROPAGATED
        self.refresh_enabled_buttons()

    def on_compute(self):
        finger_centers = extract_centers(
            self.current_mask, normalize=True, move_to_origin=True
        )
        self.finger_movements_canvas.fill(
            finger_centers, self.texts["movement_canvas_labels"]
        )
        self.finger_movements_canvas.draw()
        self.heatmap_canvas.fill(self.current_mask, self.images[0].astype(int))
        self.heatmap_canvas.draw()
        finger_movements = movement_index(self.current_mask)
        if finger_movements.get(1) is not None:
            self.console_push_text(
                self.texts["console_left_object_movement"]
                + f" {finger_movements[1]:.4f}"
            )
            if finger_movements[1] < 1.2:
                self.console_push_text(
                    self.texts["console_left_object_movement_slightly"]
                )
            elif finger_movements[1] < 2.5:
                self.console_push_text(
                    self.texts["console_left_object_movement_moderately"]
                )
            else:
                self.console_push_text(
                    self.texts["console_left_object_movement_hardly"]
                )

        if finger_movements.get(2) is not None:
            self.console_push_text(
                self.texts["console_right_object_movement"]
                + f" {finger_movements[2]:.4f}"
            )
            if finger_movements[2] < 1.2:
                self.console_push_text(
                    self.texts["console_right_object_movement_slightly"]
                )
            elif finger_movements[2] < 2.5:
                self.console_push_text(
                    self.texts["console_right_object_movement_moderately"]
                )
            else:
                self.console_push_text(
                    self.texts["console_right_object_movement_hardly"]
                )
        self.state = State.FINAL
        self.refresh_enabled_buttons()

    def on_reset(self):
        self.state = State.INITIAL
        self.refresh_enabled_buttons()
        self.reset_initial_state()

        # Reset UI with default information
        self.show_starting_image()
        self.finger_movements_canvas.clear()
        self.finger_movements_canvas.draw()
        self.heatmap_canvas.clear()
        self.heatmap_canvas.draw()
        self.lcd.setText("{: 3d} / {: 3d}".format(0, self.num_frames - 1))
        self.console.clear()
        self.console_push_text(self.texts["console_init_text"])

    def on_prev(self):
        # self.tl_slide will trigger on setValue
        self.cursur = max(0, self.cursur - 1)
        self.show_current_frame()

    def on_next(self):
        # self.tl_slide will trigger on setValue
        self.cursur = min(self.cursur + 1, self.num_frames - 1)
        self.show_current_frame()

    def on_time(self):
        self.cursur += 1
        if self.cursur > self.num_frames - 1:
            self.cursur = 0
        self.show_current_frame()

    def on_play(self):
        if self.timer.isActive():
            self.timer.stop()
            self.play_button.setText(self.texts["play_button_play_label"])
        else:
            self.timer.start(1000 // 40)
            self.play_button.setText(self.texts["play_button_stop_label"])

    def on_undo(self):
        if self.interaction is None:
            if len(self.this_frame_interactions) > 1:
                self.this_frame_interactions = self.this_frame_interactions[:-1]
                self.interacted_mask = self.this_frame_interactions[-1].predict()
            else:
                self.reset_this_interaction()
                self.interacted_mask = self.processor.prob[:, self.cursur].clone()
        else:
            if self.interaction.can_undo():
                self.interacted_mask = self.interaction.undo()
            else:
                if len(self.this_frame_interactions) > 0:
                    self.interaction = None
                    self.interacted_mask = self.this_frame_interactions[-1].predict()
                else:
                    self.reset_this_interaction()
                    self.interacted_mask = self.processor.prob[:, self.cursur].clone()

            # Update visualization
            if len(self.vis_hist) > 0:
                # Might be empty if we are undoing the entire interaction
                self.vis_map, self.vis_alpha = self.vis_hist.pop()

        # Commit changes
        self.update_interacted_mask()

    def set_navi_enable(self, boolean):
        self.run_button.setEnabled(boolean)
        self.play_button.setEnabled(boolean)
        self.undo_button.setEnabled(boolean)
        self.record_button.setEnabled(boolean)
        self.reset_button.setEnabled(boolean)

    def hit_number_key(self, number):
        if number == self.current_object:
            return
        self.current_object = number
        if self.fbrs_controller is not None:
            self.fbrs_controller.unanchor()
        self.console_push_text(self.text["console_selected_object"] + f" {number}")
        self.clear_brush()
        self.vis_brush(self.last_ex, self.last_ey)
        self.update_interact_vis()
        self.show_current_frame()

    def clear_brush(self):
        self.brush_vis_map.fill(0)
        self.brush_vis_alpha.fill(0)

    def vis_brush(self, ex, ey):
        self.brush_vis_map = cv2.circle(
            self.brush_vis_map,
            (int(round(ex)), int(round(ey))),
            self.brush_size // 2 + 1,
            color_map[self.current_object],
            thickness=-1,
        )
        self.brush_vis_alpha = cv2.circle(
            self.brush_vis_alpha,
            (int(round(ex)), int(round(ey))),
            self.brush_size // 2 + 1,
            0.5,
            thickness=-1,
        )

    def on_press(self, event):
        if self.waiting_to_start:
            self.waiting_to_start = False

        ex, ey = self.get_scaled_pos(event.x(), event.y())

        self.pressed = True
        self.right_click = event.button() != 1
        # Push last vis map into history

        self.vis_hist.append((self.vis_map.copy(), self.vis_alpha.copy()))

        if self.interaction is None:
            if len(self.this_frame_interactions) > 0:
                prev_soft_mask = self.this_frame_interactions[-1].out_prob
            else:
                prev_soft_mask = self.processor.prob[1:, self.cursur]
        else:
            # Not used if the previous interaction is still valid
            # Don't worry about stacking effects here
            prev_soft_mask = self.interaction.out_prob
        prev_hard_mask = self.processor.masks[self.cursur]
        image = self.processor.images[:, self.cursur]
        h, w = self.height, self.width

        last_interaction = self.interaction
        new_interaction = None
        if (
            last_interaction is None
            or type(last_interaction) != ClickInteraction
            or last_interaction.tar_obj != self.current_object
        ):
            self.complete_interaction()
            self.fbrs_controller.unanchor()
            new_interaction = ClickInteraction(
                image,
                prev_soft_mask,
                (h, w),
                self.fbrs_controller,
                self.current_object,
                self.processor.pad,
            )

        if new_interaction is not None:
            self.interaction = new_interaction

        # Just motion it as the first step
        self.on_motion(event)

    def on_motion(self, event):
        ex, ey = self.get_scaled_pos(event.x(), event.y())
        self.last_ex, self.last_ey = ex, ey
        self.clear_brush()
        # Visualize
        self.vis_brush(ex, ey)
        self.update_interact_vis()

    def update_interacted_mask(self):
        self.processor.update_mask_only(self.interacted_mask, self.cursur)
        self.current_mask[self.cursur] = self.processor.np_masks[self.cursur]
        self.show_current_frame()

    def complete_interaction(self):
        if self.interaction is not None:
            self.clear_visualization()
            self.interactions["annotated_frame"].append(self.cursur)
            self.interactions["interact"][self.cursur].append(self.interaction)
            self.this_frame_interactions.append(self.interaction)
            self.interaction = None
            self.undo_button.setDisabled(False)

    def on_release(self, event):
        ex, ey = self.get_scaled_pos(event.x(), event.y())
        self.console_push_text(self.texts["console_click"] + f" {self.cursur}")
        interaction = self.interaction
        ex, ey = self.get_scaled_pos(event.x(), event.y())
        self.vis_map, self.vis_alpha = interaction.push_point(
            ex, ey, self.right_click, (self.vis_map, self.vis_alpha)
        )

        self.interacted_mask = interaction.predict()
        self.update_interacted_mask()

        self.pressed = self.right_click = False
        self.undo_button.setDisabled(False)

    def refresh_enabled_buttons(self):
        self.reset_button.setEnabled(True)
        self.record_button.setEnabled(True)
        if self.state == State.INITIAL:
            self.spanish_button.setEnabled(True)
            self.english_button.setEnabled(True)
            self.play_button.setEnabled(False)
            self.run_button.setEnabled(False)
            self.undo_button.setEnabled(False)
            self.compute_button.setEnabled(False)
            self.main_canvas.setMouseTracking(False)
        elif self.state == State.RECORDED:
            self.spanish_button.setEnabled(False)
            self.english_button.setEnabled(False)
            self.play_button.setEnabled(True)
            self.run_button.setEnabled(True)
            self.undo_button.setEnabled(False)
            self.compute_button.setEnabled(False)
            self.main_canvas.setMouseTracking(True)
        elif self.state == State.PROPAGATED:
            self.play_button.setEnabled(True)
            self.run_button.setEnabled(True)
            self.undo_button.setEnabled(False)
            self.compute_button.setEnabled(True)
            self.main_canvas.setMouseTracking(True)
        elif self.state == State.FINAL:
            self.play_button.setEnabled(True)
            self.run_button.setEnabled(False)
            self.undo_button.setEnabled(False)
            self.compute_button.setEnabled(False)
            self.main_canvas.setMouseTracking(True)


if __name__ == "__main__":

    # Arguments parsing
    parser = ArgumentParser()
    parser.add_argument("--prop_model", default="saves/stcn.pth")
    parser.add_argument("--fusion_model", default="saves/fusion_stcn.pth")
    parser.add_argument("--s2m_model", default="saves/s2m_mobilenet.pth")
    parser.add_argument("--fbrs_model", default="saves/fbrs.pth")
    parser.add_argument(
        "--starting-image",
        help="Placeholder image to show when the program is idle.",
        default="../Mascaras/P1/Visita_1_OFF/Dedos_enfrentados/first_frame.png",
    )
    parser.add_argument(
        "--n_frames",
        help="Number of frames to record in each execution.",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--num_objects",
        help="Default: 1 if no masks provided, masks.max() otherwise",
        default=2,
        type=int,
    )
    parser.add_argument("--mem_freq", default=10, type=int)
    parser.add_argument(
        "--mem_profile",
        default=2,
        type=int,
        help="0 - Faster and more memory intensive; 2 - Slower and less memory intensive. Default: 0.",
    )
    parser.add_argument("--no_amp", help="Turn off AMP", action="store_true")
    parser.add_argument(
        "--resolution", help="Pass -1 to use original size", default=400, type=int
    )
    args = parser.parse_args()

    with torch.cuda.amp.autocast(enabled=not args.no_amp):

        # Load our checkpoint
        prop_saved = torch.load(args.prop_model)
        prop_model = PropagationNetwork().cuda().eval()
        prop_model.load_state_dict(prop_saved)

        fusion_saved = torch.load(args.fusion_model)
        fusion_model = FusionNet().cuda().eval()
        fusion_model.load_state_dict(fusion_saved)

        # Loads the S2M model
        if args.s2m_model is not None:
            s2m_saved = torch.load(args.s2m_model)
            s2m_model = S2M().cuda().eval()
            s2m_model.load_state_dict(s2m_saved)
        else:
            s2m_model = None

        # Load the placeholder image
        starting_image = cv2.imread(args.starting_image)
        starting_image = cv2.cvtColor(starting_image, cv2.COLOR_BGR2RGB)

        # Determine the number of objects
        num_objects = args.num_objects
        if num_objects is None:
            if masks is not None:
                num_objects = masks.max()
            else:
                num_objects = 1

        s2m_controller = S2MController(s2m_model, num_objects, ignore_class=255)
        if args.fbrs_model is not None:
            fbrs_controller = FBRSController(args.fbrs_model)
        else:
            fbrs_controller = None

        app = QApplication(sys.argv)
        ex = App(
            prop_model,
            fusion_model,
            s2m_controller,
            fbrs_controller,
            starting_image,
            args.n_frames,
            num_objects,
            args.mem_freq,
            args.mem_profile,
        )
        apply_stylesheet(
            app, theme=os.path.join("assets", "dasci_colors.xml"), invert_secondary=True
        )
        stylesheet = app.styleSheet()
        with open(os.path.join("assets", "custom.css"), "r") as f:
            app.setStyleSheet(stylesheet + f.read().format(**os.environ))

        sys.exit(app.exec_())
