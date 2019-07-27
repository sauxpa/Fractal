#!/usr/bin/env python
# coding: utf-8

# all credit to https://www.datacamp.com/community/tutorials/matplotlib-3d-volumetric-data

import numpy as np
import matplotlib.pyplot as plt

class multi_slice_viewer():
    def __init__(self, axis=0, volume=None):
        self._axis = axis
        self._volume = volume

    @property
    def axis(self):
        return self._axis
    @axis.setter
    def axis(self, new_axis):
        self._axis = new_axis

    @property
    def volume(self):
        return self._volume
    @volume.setter
    def volume(self, new_volume):
        self._volume = new_volume

    def remove_keymap_conflicts(self, new_keys_set):
        for prop in plt.rcParams:
            if prop.startswith('keymap.'):
                keys = plt.rcParams[prop]
                remove_list = set(keys) & new_keys_set
                for key in remove_list:
                    keys.remove(key)

    def render(self, fig, ax):
        self.fig, self.ax = fig, ax
        self.remove_keymap_conflicts({'j', 'k'})
        self.ax.volume = self.volume
        self.ax.index = self.volume.shape[0] // 2 # start at the center
        self.ax.imshow(np.take(self.volume, self.ax.index, self.axis))
        self.fig.canvas.mpl_connect('key_press_event', self.process_key)
        plt.axis('off')
        plt.show()

    def process_key(self, event):
        """Use 'j' and 'k' keys to navigate the 2d slices of the volume
        """
        fig = event.canvas.figure
        ax = fig.axes[0]
        if event.key == 'j':
            self.previous_slice()
        elif event.key == 'k':
            self.next_slice()
        fig.canvas.draw()

    def previous_slice(self):
        volume = self.ax.volume
        self.ax.index = (self.ax.index - 1) % volume.shape[0]  # wrap around using %
        self.ax.images[0].set_array(np.take(volume, self.ax.index, axis=self.axis))

    def next_slice(self):
        volume = self.ax.volume
        self.ax.index = (self.ax.index + 1) % volume.shape[0]
        self.ax.images[0].set_array(np.take(volume, self.ax.index, axis=self.axis))