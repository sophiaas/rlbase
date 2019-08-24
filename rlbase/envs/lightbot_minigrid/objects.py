from gym_minigrid.minigrid import WorldObj

class Light(WorldObj):
    def __init__(self, color='blue'):
        super().__init__('light', color)
        self.is_on = False

    def can_overlap(self):
        return True

    def render(self, r):
        self._set_color(r)
        r.drawPolygon([
            (0          , CELL_PIXELS),
            (CELL_PIXELS, CELL_PIXELS),
            (CELL_PIXELS,           0),
            (0          ,           0)
        ])

    def toggle(self):
        if self.is_on:
            return False
        else:
            self.is_on = not self.is_on
            self.color = 'yellow'
            return True