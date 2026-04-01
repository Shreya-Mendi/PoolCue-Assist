"""
Shot advisor using Duke OIT LiteLLM proxy (OpenAI-compatible endpoint).
https://litellm.oit.duke.edu/

Set DUKE_API_KEY env var or write key to ~/.duke_litellm_key.
Model is configurable in config/settings.json (default: gpt-4o-mini).
"""

import os
from pathlib import Path
from openai import OpenAI


DUKE_LITELLM_BASE = "https://litellm.oit.duke.edu"


def _load_api_key() -> str:
    key = os.environ.get("DUKE_API_KEY")
    if key:
        return key
    key_file = Path.home() / ".duke_litellm_key"
    if key_file.exists():
        return key_file.read_text().strip()
    raise RuntimeError(
        "No Duke LiteLLM API key found. "
        "Set DUKE_API_KEY env var or write key to ~/.duke_litellm_key"
    )


class ShotAdvisor:
    def __init__(self, model="gpt-4o-mini"):
        self.model = model
        self.client = OpenAI(
            api_key=_load_api_key(),
            base_url=DUKE_LITELLM_BASE,
        )
        self._last_recommendation = None

    def recommend(self, balls: dict, pockets: dict, game_state: dict) -> dict:
        """
        Ask LLM for the best shot given current table state.

        Args:
            balls: {"cue": (cx,cy), "4": (cx,cy), ...}
            pockets: {"top-left": (cx,cy), ...}
            game_state: {"mode": "8ball", "player_type": "solids", "frame_w": 640, "frame_h": 480}

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

        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=200,
            temperature=0.2,
            messages=[
                {"role": "system", "content": "You are a billiards coach giving shot recommendations. Follow the exact output format requested."},
                {"role": "user", "content": prompt}
            ]
        )

        return self._parse_response(response.choices[0].message.content)

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
