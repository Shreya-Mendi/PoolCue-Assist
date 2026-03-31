"""
Game state tracker for 8-ball pool.
Tracks which balls are on the table, whose turn it is, and player types.
Detects when a ball is sunk by comparing detected balls between frames.
Announces events via speaker.
"""

SOLIDS = {str(i) for i in range(1, 8)}    # 1-7
STRIPES = {str(i) for i in range(9, 16)}  # 9-15


class GameState:
    def __init__(self, speaker=None, mode="8ball"):
        self.speaker = speaker
        self.mode = mode

        # Which balls are still on the table
        self.balls_on_table: set = SOLIDS | STRIPES | {"8"}

        # Player assignment: 0 = unassigned, "solids", "stripes"
        self.player_types = {1: None, 2: None}
        self.current_player = 1

        # Shot history for analysis
        self.shot_log = []

    def update(self, detected_balls: dict):
        """
        Compare newly detected balls to known table state.
        Announce any balls that have been sunk.
        Returns set of ball labels currently on table.
        """
        detected_labels = set(detected_balls.keys()) - {"cue"}
        sunk = self.balls_on_table - detected_labels

        for ball in sunk:
            self.balls_on_table.discard(ball)
            if self.speaker:
                self.speaker.say(f"{ball} ball sunk")

        return self.balls_on_table.copy()

    def remaining_for_player(self) -> set:
        """Return the balls the current player still needs to sink."""
        ptype = self.player_types[self.current_player]
        if ptype == "solids":
            remaining = self.balls_on_table & SOLIDS
            if not remaining:
                return {"8"}
            return remaining
        elif ptype == "stripes":
            remaining = self.balls_on_table & STRIPES
            if not remaining:
                return {"8"}
            return remaining
        # Unassigned: any ball except 8
        return (self.balls_on_table & (SOLIDS | STRIPES))

    def assign_player(self, player: int, ptype: str):
        """Assign 'solids' or 'stripes' to a player."""
        self.player_types[player] = ptype
        other = 2 if player == 1 else 1
        self.player_types[other] = "stripes" if ptype == "solids" else "solids"
        if self.speaker:
            self.speaker.say(f"Player {player} shoots {ptype}")

    def next_turn(self):
        self.current_player = 2 if self.current_player == 1 else 1
        if self.speaker:
            self.speaker.say(f"Player {self.current_player}'s turn")

    def log_shot(self, recommendation: dict, stroke_label: str, aim_error_px: float):
        self.shot_log.append({
            "player": self.current_player,
            "ball": recommendation.get("ball"),
            "pocket": recommendation.get("pocket"),
            "difficulty": recommendation.get("difficulty"),
            "stroke": stroke_label,
            "aim_error_px": aim_error_px,
        })

    def check_game_over(self) -> bool:
        """Return True if 8-ball has been sunk."""
        return "8" not in self.balls_on_table

    @property
    def current_player_type(self) -> str:
        return self.player_types[self.current_player] or "any"
