"""
pg_image_tools -- reusable interactive 2D image plotting for PyQt5 + pyqtgraph
==============================================================================

A drop-in interactive image viewer: click-to-measure, live cursor readout, a
lineout panel with a 25%-75% resolution metric, display filters, and a populated
right-click menu. It knows nothing about where your data comes from -- you hand
it a numpy array and a pixel size, it renders and measures.

This package is self-contained. It imports only ``numpy``, ``scipy.ndimage``,
``PyQt5`` and ``pyqtgraph``, and nothing from whatever project it lives in.


Requirements
------------
numpy, scipy, PyQt5, pyqtgraph


Quick start
-----------
The only thing you need is somewhere to put it::

    from pg_image_tools import ImagePlotWidget

    plot = ImagePlotWidget(self.some_placeholder_widget)
    plot.set_image(my_2d_array, pixel_size_m=8.7e-9)

``ImagePlotWidget`` is a ``QWidget``, so ``container`` is optional -- omit it and
add the widget to a layout yourself, or ``show()`` it as a top-level window.


Full example (Qt Designer host)
-------------------------------
Every companion widget is optional. Supply the ones you have; the features you
skip simply do not appear::

    class MyWindow(QtWidgets.QMainWindow):
        def __init__(self):
            super().__init__()
            uic.loadUi("my_window.ui", self)

            self.plot = ImagePlotWidget(
                self.plotContainer,                    # a plain QWidget from the .ui
                info_label=self.label_plot_info,       # optional
                title_label=self.label_plot_title,     # optional
                transpose_checkbox=self.checkBox_transpose,   # optional
                log_checkbox=self.checkBox_logCmap,           # optional
            )

            # Add your own entries to the image's right-click menu
            self.plot.add_menu_action("Copy File Path", self._copy_path, at_top=True)
            self.act_full = self.plot.add_menu_action(
                "Full Probe Zoom", checkable=True,
                after=self.plot.action_auto_reset_zoom,
            )

        def show_array(self, arr, pixel_size_m, caption):
            self.plot.set_title(caption)               # no-op without a title_label
            self.plot.set_image(arr, pixel_size_m=pixel_size_m)

Companions can also be bound after construction with ``attach_info_label()``,
``attach_title_label()``, ``attach_transpose_checkbox()`` and
``attach_log_checkbox()``.


Optional widgets
----------------
================== ============================================ ==========================
argument           when supplied                                when omitted
================== ============================================ ==========================
info_label         shows "W×H µm, N nm pix" plus a live          no readout; cursor
                   ``x=, y=, I=`` line as the cursor moves       position still emitted
                                                                 via sigMouseMovedImage
title_label        set_title() writes into it, each line         set_title() is a no-op
                   elided in the middle, full text as tooltip
transpose_checkbox image is transposed before display;           image never transposed
                   toggling redraws immediately
log_checkbox       displays log10(|data|); toggling redraws      linear display only
================== ============================================ ==========================

Other constructor flags: ``enable_lineout``, ``enable_measure``,
``enable_filters`` (all default True) and ``title_elide_width`` (default 500 px).


Interactions
------------
Image
    * Left-click two points -> cross markers, a dashed connecting line, and a
      label with the separation in µm and in pixels. A third click clears them.
    * Right-click menu: ``Plot Lineout``, ``Auto-reset Zoom`` (re-fit the view on
      every new image), and ``Analyze > Median Filter`` / ``Gaussian Filter``
      (each prompts for a kernel width). Zooming back out is pyqtgraph's own
      ``View All``.

Lineout panel (shown by ``Plot Lineout``)
    * Plots the image sampled along the two measurement points.
    * Two draggable vertical cursors, blue and green, continuously report their
      x positions, the values under them, and ΔX.
    * Drag them to either side of a hard edge, then right-click the lineout and
      choose ``Find 25%-75% resolution``: the crossing width is shaded and
      reported in nm.


Data flow and caching
---------------------
``set_image()`` keeps a reference to the raw array. Display-option changes
(filter, transpose, log) re-render from that cache -- the widget never asks the
host to reload, and the current zoom, measurement points, lineout and scatter
overlay survive the redraw. Toggling transpose is the one exception: it changes
the coordinate system, so the pixel-space overlays are cleared.

``sigRedrawRequested`` fires *after* such a redraw. It is a notification for
hosts that keep their own derived state -- not a request for data. Leave it
unconnected if you have none.


Coordinate conventions
----------------------
* Images follow the pyqtgraph convention: axis 0 is x, axis 1 is y. Pass arrays
  already in that orientation (or use the transpose checkbox).
* ``pixel_size_m`` is metres per pixel and drives every µm / nm readout.
  Leave it at 1.0 and the readouts are effectively in pixels × 1e6.
* Measurement points and ``set_scatter_overlay(x, y)`` are in image pixel
  coordinates, origin at the array corner. The cursor readout reports µm
  relative to the image *centre*.


Extension points
----------------
``add_menu_action(text, callback, checkable=, checked=, at_top=, after=)``
    Add an entry to the image's right-click menu and get the QAction back.
    ``at_top=True`` places it above this widget's own block (still below
    pyqtgraph's built-ins); ``after=some_action`` places it directly after an
    existing entry. Also ``add_menu_separator()`` and ``add_menu_submenu()``.

``set_scatter_overlay(x, y)`` / ``clear_scatter_overlay()``
    A red scatter layer in image coordinates for host-supplied points.

``zoom_to_left_square()``, ``reset_zoom()``, ``set_view_range(**kwargs)``
    Zoom helpers. ``reset_zoom()`` is the programmatic form of "View All"; the
    last is a passthrough to ``ViewBox.setRange``.

``image_view``, ``view``, ``image_item``, ``menu``, ``lineout``
    Escape hatches to the underlying pyqtgraph objects for anything not covered.

``displayed_shape``, ``pixel_size_m``, ``raw_data``, ``measure_points``, ``info_text``
    Read-only state.

Signals: ``sigRedrawRequested()``, ``sigPointsChanged(list)``,
``sigMouseMovedImage(float, float)``.


Demo
----
``python -m pg_image_tools`` opens a standalone window on a synthetic test
pattern with all optional widgets wired up. Useful as a smoke test after edits.


Packaging
---------
This folder has no imports from its host project, so it can be moved into its
own repository or turned into a pip package by itself. Keep the top-level module
name ``pg_image_tools`` and ``from pg_image_tools import ImagePlotWidget`` keeps
working unchanged in every dependent project.
"""

from ._image_view import ImagePlotWidget
from ._lineout import LineoutPanel, compute_lineout

__all__ = ["ImagePlotWidget", "LineoutPanel", "compute_lineout"]
__version__ = "0.1.0"
