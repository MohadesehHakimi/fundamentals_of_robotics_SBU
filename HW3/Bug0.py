from base_bug import (BaseBugRobotController, dist, normalize_angle,
                      TARGET_WALL_DISTANCE_SIDE, WALL_FAR_DISTANCE_THRESHOLD,
                      ALIGNMENT_TOLERANCE_RADIANS)


class Bug0RobotController(BaseBugRobotController):
    def __init__(self):
        super().__init__()
        self.hit_point = None
        # Override sensor groupings from the base class
        self.front_ids = (0, 7)  # ps0, ps7
        self.right_id = 2  # ps2
        self.left_id = 5  # ps5

    def _wall_follow(self, ds, side='left'):
        """
        Overrides the base class's wall-following method with single side sensors.
        """
        # Front obstacle: immediate avoidance
        if self._front_blocked(ds):
            if side == 'left':
                self._set_speed(0.25, -0.25)  # pivot right
            else:
                self._set_speed(-0.25, 0.25)  # pivot left
            return

        # Distance to wall on the selected side
        wall_dist = ds[self.left_id if side == 'left' else self.right_id]

        # Lost the wall? – gentle turn toward where it should be
        if wall_dist < 0 or wall_dist > WALL_FAR_DISTANCE_THRESHOLD:
            if side == 'left':
                self._set_speed(0.15, 0.30)  # curve left
            else:
                self._set_speed(0.30, 0.15)  # curve right
            return

        # Proportional controller around TARGET_WALL_DISTANCE_SIDE
        error = TARGET_WALL_DISTANCE_SIDE - wall_dist  # positive => too close
        Kp = 1.5
        corr = Kp * error

        # Sign flips depending on side
        sign = 1 if side == 'left' else -1
        left = 0.40 - sign * corr
        right = 0.40 + sign * corr

        # Clamp motor factors to [-1, 1]
        self._set_speed(max(-1, min(1, left)), max(-1, min(1, right)))

    def _bug_step(self, ds):
        """Implements the Bug0 algorithm state machine."""
        if self.bug_state == "GOAL_SEEK":
            if self._align(self._angle_to(self.goal)):
                if self._front_blocked(ds):
                    self.hit_point = self.pos
                    self.bug_state = "WALL_FOLLOW"
                    self._set_speed(-0.25, 0.25)
                else:
                    self._set_speed(0.5, 0.5)

        elif self.bug_state == "WALL_FOLLOW":
            # Check if we can leave the wall
            ang_to_goal = self._angle_to(self.goal)
            facing_goal = abs(normalize_angle(ang_to_goal - self.th)) < ALIGNMENT_TOLERANCE_RADIANS * 2.5
            if (facing_goal and
                    not self._front_blocked(ds) and
                    dist(self.pos, self.goal) < dist(self.hit_point, self.goal) - 0.05):
                self.bug_state = "GOAL_SEEK"
                self.hit_point = None
                return

            self._wall_follow(ds, side='left')


if __name__ == "__main__":
    print("Starting Bug-0 controller…")
    controller = Bug0RobotController()
    controller.run()
