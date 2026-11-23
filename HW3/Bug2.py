from base_bug import BaseBugRobotController, dist, START_POS, M_LINE_TOLERANCE_BUG2


class Bug2RobotController(BaseBugRobotController):
    def __init__(self):
        super().__init__()
        self.hit_point = None
        self.m_line_start = START_POS
        self.front_ids = (0, 7, 1, 6)  # Overriding front sensors for Bug2

    @staticmethod
    def _point_line_distance_sq(p, a, b):
        """Calculates the squared distance from a point to a line segment."""
        abx, aby = b[0] - a[0], b[1] - a[1]
        ab2 = abx * abx + aby * aby
        if ab2 < 1e-12:
            return (p[0] - a[0]) ** 2 + (p[1] - a[1]) ** 2
        area2 = abs((p[0] - a[0]) * aby - (p[1] - a[1]) * abx) ** 2
        return area2 / ab2

    def _on_m_line(self):
        """Checks if the robot is on the m-line."""
        d2 = self._point_line_distance_sq(self.pos, self.m_line_start, self.goal)
        if d2 > M_LINE_TOLERANCE_BUG2 ** 2:
            return False

        sgx, sgy = self.goal[0] - self.m_line_start[0], self.goal[1] - self.m_line_start[1]
        spx, spy = self.pos[0] - self.m_line_start[0], self.pos[1] - self.m_line_start[1]
        dot = sgx * spx + sgy * spy
        seg_len2 = sgx * sgx + sgy * sgy
        if seg_len2 < 1e-12:
            return True
        proj = dot / seg_len2
        return -0.05 < proj < 1.05

    def _bug_step(self, ds):
        """Implements the Bug2 algorithm logic."""
        if self.bug_state == "GOAL_SEEK":
            angle_to_goal = self._angle_to(self.goal)
            if self._align(angle_to_goal):
                if self._front_blocked(ds):
                    self.bug_state = "WALL_FOLLOW"
                    self.hit_point = self.pos
                    self._set_speed(-0.25, 0.25)
                else:
                    self._set_speed(0.6, 0.6)

        elif self.bug_state == "WALL_FOLLOW":
            if self.hit_point and self._on_m_line():
                if (dist(self.pos, self.goal) <
                        dist(self.hit_point, self.goal) - 0.05 and
                        dist(self.pos, self.hit_point) > M_LINE_TOLERANCE_BUG2 * 1.5):
                    self.bug_state = "GOAL_SEEK"
                    self.hit_point = None
                    return
            self._wall_follow(ds, side='left')


if __name__ == "__main__":
    print("Starting Bug-2 controller…")
    Bug2RobotController().run()
