from base_bug import BaseBugRobotController, dist, HIT_POINT_RETURN_TOLERANCE


class Bug1RobotController(BaseBugRobotController):
    def __init__(self):
        super().__init__()
        self.hit_point = None
        self.leave_point = None
        self.best_dist = float('inf')
        self.made_loop = False
        self.leave_timer = 0
        self.front_ids = (0, 7, 1, 6)  # Overriding front sensors for Bug1

    def _bug_step(self, ds):
        """Implements the Bug1 algorithm logic."""
        if self.bug_state == "GOAL_SEEK":
            ang = self._angle_to(self.goal)
            if self._align(ang):
                if self._front_blocked(ds):
                    self.bug_state = "WALL_FOLLOW"
                    self.hit_point = self.pos
                    self.leave_point = self.pos
                    self.best_dist = dist(self.pos, self.goal)
                    self.made_loop = False
                    self._set_speed(-0.25, 0.25)
                else:
                    self._set_speed(0.6, 0.6)

        elif self.bug_state == "WALL_FOLLOW":
            if not self.made_loop:
                d2goal = dist(self.pos, self.goal)
                if d2goal < self.best_dist - 0.01:
                    self.best_dist = d2goal
                    self.leave_point = self.pos

            if (not self.made_loop and
                    dist(self.pos, self.hit_point) < HIT_POINT_RETURN_TOLERANCE and
                    dist(self.pos, self.leave_point) > 0.1):
                self.made_loop = True

            if self.made_loop and dist(self.pos, self.leave_point) < HIT_POINT_RETURN_TOLERANCE:
                self.bug_state = "LEAVING_WALL"
                self.leave_timer = 20
                return

            self._wall_follow(ds, side='left')

        elif self.bug_state == "LEAVING_WALL":
            if self.leave_timer > 0:
                self._set_speed(0.8, 0.4)
                self.leave_timer -= 1
            else:
                self._stop()
                self.bug_state = "GOAL_SEEK"


if __name__ == "__main__":
    print("Starting Bug-1 controller…")
    Bug1RobotController().run()
