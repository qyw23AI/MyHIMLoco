"""Mostly flat terrain with bounded slopes/roughness; no stairs or gaps."""
import numpy as np
from ..terrain.rough import Terrain
from ..terrain import heightfield


class GentleTerrain(Terrain):
    def make_terrain(self, choice, difficulty):
        tile = heightfield.SubTerrain('gentle', width=self.width_per_env_pixels,
                                     length=self.length_per_env_pixels,
                                     vertical_scale=self.cfg.vertical_scale,
                                     horizontal_scale=self.cfg.horizontal_scale)
        if choice < 0.6:
            return tile
        if choice < 0.8:
            sign = -1 if np.random.rand() < 0.5 else 1
            heightfield.pyramid_sloped_terrain(tile, slope=sign * 0.12 * difficulty,
                                              platform_size=2.0)
        else:
            amplitude = 0.005 + 0.015 * difficulty
            heightfield.random_uniform_terrain(tile, -amplitude, amplitude,
                                               step=self.cfg.vertical_scale,
                                               downsampled_scale=0.3)
        return tile
