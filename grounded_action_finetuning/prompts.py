"""The same decision/action prompts are used for training and rollout."""

from prismatic.models.backbones.llm.prompting import PurePromptBuilder


def action_question(instruction: str, side: str) -> str:
    side = side.strip().lower()
    if side not in {"left", "right"}:
        raise ValueError(f"Expected left or right, got {side!r}")
    return (f"What action should the robot take to {instruction.lower()}? "
            f"The correct destination is on the {side}.")


def action_prompt(instruction: str, side: str, action_tokens: str | None = None) -> str:
    builder = PurePromptBuilder("openvla")
    builder.add_turn("human", action_question(instruction, side))
    if action_tokens is not None:
        builder.add_turn("gpt", action_tokens)
    return builder.get_prompt()
