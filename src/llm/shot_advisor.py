"""
Shot advisor using Claude API.
Takes current table state (ball positions, pocket positions, game context)
and returns a structured shot recommendation with spoken text.
"""

import os
import anthropic
from pathlib import Path


def _load_api_key() -> str:
    # Check env var first, then fallback to file
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    key_file = Path.home() / ".anthropic_key"
    if key_file.exists():
        return key_file.read_text().strip()
    raise RuntimeError(
        "No Anthropic API key found. "
        "Set ANTHROPIC_API_KEY env var or write key to ~/.anthropic_key"
    )


class ShotAdvisor:
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=_load_api_key())
        self._last_recommendation = None

    def recommend(self, balls: dict, pockets: dict, game_state: dict) -> dict:
        """
        Ask Claude for the best shot given current table state.

        Args:
            balls: {"cue": (cx,cy), "4": (cx,cy), ...}
            pockets: {"top-left": (cx,cy), ...}
            game_state: {"mode": "8ball", "player_type": "solids", "turn": 1}

        Returns:
            {
                "ball": "4",
                "pocket": "top-right",
                "spoken": "Hit the 4 ball into the top-right pocket.",
                "reason": "Clear straight line, no blockers.",
                "difficulty": 2
            }
        """
        prompt = self._build_prompt(balls, pockets, game_state)

        message = self.client.messages.create(
            model="claude-haiku-4-5-20251001",  # Haiku: fast + cheap for real-time use
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}]
        )

        return self._parse_response(message.content[0].text)

    def _build_prompt(self, balls, pockets, game_state) -> str:
        mode = game_state.get("mode", "8ball")
        player_type = game_state.get("player_type", "any")
        frame_w = game_state.get("frame_w", 640)
        frame_h = game_state.get("frame_h", 480)

        ball_lines = "\n".join(
            f"  Ball {label}: pixel ({cx}, {cy})"
            for label, (cx, cy) in sorted(balls.items())
            if label != "cue"
        )
        cue_pos = balls.get("cue", "unknown")
        pocket_lines = "\n".join(
            f"  {name}: pixel ({cx}, {cy})"
            for name, (cx, cy) in pockets.items()
        )

        return f"""You are a billiards coach analyzing a {mode} game from an overhead camera.
Frame size: {frame_w}x{frame_h} pixels (origin top-left).

Cue ball position: {cue_pos}

Remaining balls on table:
{ball_lines}

Pocket positions:
{pocket_lines}

Player shoots: {player_type}
{"(Solids = balls 1-7, Stripes = 9-15, 8-ball last)" if mode == "8ball" else ""}

Choose the single best shot. Consider: straight lines, no obstructing balls in path, angle to pocket.
Reply ONLY in this exact format (no extra text):
BALL: [ball number or label]
POCKET: [pocket name]
SPOKEN: [one natural sentence a coach would say, max 12 words]
REASON: [one sentence explanation]
DIFFICULTY: [1=easy to 5=very hard]"""

    def _parse_response(self, text: str) -> dict:
        result = {
            "ball": None,
            "pocket": None,
            "spoken": "I recommend your next shot.",
            "reason": "",
            "difficulty": 3
        }
        for line in text.strip().splitlines():
            if line.startswith("BALL:"):
                result["ball"] = line.split(":", 1)[1].strip()
            elif line.startswith("POCKET:"):
                result["pocket"] = line.split(":", 1)[1].strip()
            elif line.startswith("SPOKEN:"):
                result["spoken"] = line.split(":", 1)[1].strip()
            elif line.startswith("REASON:"):
                result["reason"] = line.split(":", 1)[1].strip()
            elif line.startswith("DIFFICULTY:"):
                try:
                    result["difficulty"] = int(line.split(":", 1)[1].strip())
                except ValueError:
                    pass
        self._last_recommendation = result
        return result

    @property
    def last(self) -> dict | None:
        return self._last_recommendation
