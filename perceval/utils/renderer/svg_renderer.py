# MIT License
#
# Copyright (c) 2022 Quandela
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

from .generic_renderer import Renderer, Canvas

import drawSvg as draw


class SVGCanvas(Canvas):
    def __init__(self, render_size=1, **opts):
        super().__init__(**opts, inverse_Y=True)
        self._draws = []
        self._render_size = render_size

    def add_mline(self, points, stroke="black", stroke_width=1, stroke_linejoin="miter",
                  stroke_dasharray=None):
        points = super().add_mline(points, stroke, stroke_width)
        self._draws.append(draw.Lines(*points, stroke=stroke, stroke_width=stroke_width,
                                      fill="none", close=False))

    def add_polygon(self, points, stroke="black", stroke_width=1, fill=None, stroke_linejoin="miter",
                    stroke_dasharray=None):
        points = super().add_polygon(points, stroke, stroke_width, fill)
        if fill is None:
            fill = "none"
        self._draws.append(draw.Lines(*points, stroke=stroke, fill=fill, close=True,
                                      stroke_dasharray=stroke_dasharray,
                                      stroke_linejoin=stroke_linejoin))

    def add_mpath(self, points, stroke="black", stroke_width=1, fill=None, stroke_linejoin="miter",
                  stroke_dasharray=None):
        points = super().add_mpath(points, stroke, stroke_width, fill)
        if fill is None:
            fill = "none"
        p = draw.Path(stroke_width=stroke_width, stroke=stroke, stroke_linejoin=stroke_linejoin,
                      fill=fill)
        idx = 0
        while idx < len(points):
            if points[idx] == 'M':
                p.M(*points[idx+1:idx+3])
                idx += 2
            elif points[idx] == 'L':
                p.L(*points[idx + 1:idx + 3])
                idx += 2
            elif points[idx] == 'S':
                p.S(*points[idx + 1:idx + 5])
                idx += 4
            elif points[idx] == 'C':
                p.C(*points[idx+1:idx+7])
                idx += 6
            idx += 1
        self._draws.append(p)

    def add_circle(self, points, r, stroke="black", stroke_width=1, fill=None,
                   stroke_dasharray=None):
        points = super().add_circle(points, r, stroke, stroke_width, fill)
        if fill is None:
            fill = "none"
        self._draws.append(draw.Circle(points[0], points[1], r,
                                       stroke_width=stroke_width, fill=fill, stroke=stroke))

    def add_text(self, points, text, size, ta="start"):
        if ta == "right":
            ta = "end"
        elif ta == "left":
            ta = "start"
        points = super().add_text(points, text, size, ta)
        self._draws.append(draw.Text(text, size, *points, text_anchor=ta))

    def draw(self):
        super().draw()
        d = draw.Drawing(self._maxx-self._miny, self._maxy-self._miny,
                         origin=(self._minx, -self._maxy))
        for dr in self._draws:
            d.append(dr)
        return d.setPixelScale(self._render_size)


class SVGRenderer(Renderer):
    def new_canvas(self, **opts) -> Canvas:
        return SVGCanvas(**opts)
