from typing import List
from openai import OpenAI
from langfuse import Langfuse

from .config import (
    LITELLM_BASE_URL, LITELLM_API_KEY, LITELLM_MODEL,
    LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST,
)
from .models import DebateTurn

langfuse = Langfuse(
    public_key=LANGFUSE_PUBLIC_KEY,
    secret_key=LANGFUSE_SECRET_KEY,
    host=LANGFUSE_HOST,
)

llm_client = OpenAI(
    base_url=f"{LITELLM_BASE_URL}/v1",
    api_key=LITELLM_API_KEY,
)

SYSTEM_PROMPT_KO = {
    "pro": (
        "당신은 토론에서 '{topic}' 주제에 대해 **찬성** 입장을 맡은 토론자입니다.\n"
        "상대방의 주장에 논리적으로 반박하고, 자신의 입장을 근거를 들어 주장하세요.\n"
        "2-3문단 이내로 간결하게 작성하세요."
    ),
    "con": (
        "당신은 토론에서 '{topic}' 주제에 대해 **반대** 입장을 맡은 토론자입니다.\n"
        "상대방의 주장에 논리적으로 반박하고, 자신의 입장을 근거를 들어 주장하세요.\n"
        "2-3문단 이내로 간결하게 작성하세요."
    ),
}

SYSTEM_PROMPT_EN = {
    "pro": (
        "You are a debater arguing **FOR** the topic: '{topic}'.\n"
        "Logically counter the opponent's arguments and support your position with evidence.\n"
        "Keep your response concise, within 2-3 paragraphs."
    ),
    "con": (
        "You are a debater arguing **AGAINST** the topic: '{topic}'.\n"
        "Logically counter the opponent's arguments and support your position with evidence.\n"
        "Keep your response concise, within 2-3 paragraphs."
    ),
}

FIRST_TURN_KO = "'{topic}' 주제에 대해 {stance} 입장에서 첫 번째 주장을 펼쳐주세요."
REPLY_TURN_KO = "상대방({opponent_stance}측)의 주장:\n{opponent_msg}\n\n이에 대해 반박하고 {stance} 입장을 이어가세요."

FIRST_TURN_EN = "Please present your opening argument {stance} the topic: '{topic}'."
REPLY_TURN_EN = "Opponent's ({opponent_stance}) argument:\n{opponent_msg}\n\nCounter this and continue your {stance} position."


def _get_prompts(language: str):
    if language == "en":
        return SYSTEM_PROMPT_EN, FIRST_TURN_EN, REPLY_TURN_EN
    return SYSTEM_PROMPT_KO, FIRST_TURN_KO, REPLY_TURN_KO


def run_debate(topic: str, rounds: int, language: str = "ko") -> tuple[List[DebateTurn], str]:
    system_prompts, first_turn_tmpl, reply_turn_tmpl = _get_prompts(language)

    stance_labels = {
        "ko": {"pro": "찬성", "con": "반대"},
        "en": {"pro": "for", "con": "against"},
    }
    labels = stance_labels.get(language, stance_labels["ko"])

    turns: List[DebateTurn] = []
    history_pro: list = []
    history_con: list = []

    pro_system = system_prompts["pro"].format(topic=topic)
    con_system = system_prompts["con"].format(topic=topic)

    # Root span acts as the trace
    with langfuse.start_as_current_span(
        name="multi-agent-debate",
        input={"topic": topic, "rounds": rounds, "language": language},
    ) as root_span:
        root_span.update_trace(tags=["debate", language])

        for round_num in range(1, rounds + 1):
            # --- PRO turn ---
            with root_span.start_as_current_span(
                name=f"round-{round_num}-pro",
                input={"round": round_num, "agent": "pro"},
            ) as pro_span:
                pro_messages = []
                if round_num == 1:
                    user_msg = pro_system + "\n\n" + first_turn_tmpl.format(topic=topic, stance=labels["pro"])
                else:
                    last_con_msg = turns[-1].message
                    user_msg = reply_turn_tmpl.format(
                        opponent_stance=labels["con"],
                        opponent_msg=last_con_msg,
                        stance=labels["pro"],
                    )

                pro_messages.extend(history_pro)
                pro_messages.append({"role": "user", "content": user_msg})

                with pro_span.start_as_current_generation(
                    name=f"llm-pro-round-{round_num}",
                    model=LITELLM_MODEL,
                    input=pro_messages,
                ) as pro_gen:
                    pro_response = llm_client.chat.completions.create(
                        model=LITELLM_MODEL,
                        messages=pro_messages,
                        max_tokens=512,
                        temperature=0.7,
                    )
                    pro_text = pro_response.choices[0].message.content
                    pro_gen.update(
                        output=pro_text,
                        usage_details={
                            "input": pro_response.usage.prompt_tokens,
                            "output": pro_response.usage.completion_tokens,
                        },
                    )

                pro_span.update(output={"message": pro_text})
                turns.append(DebateTurn(
                    round=round_num, agent="pro", stance=labels["pro"], message=pro_text,
                ))
                history_pro.append({"role": "user", "content": user_msg})
                history_pro.append({"role": "assistant", "content": pro_text})

            # --- CON turn ---
            with root_span.start_as_current_span(
                name=f"round-{round_num}-con",
                input={"round": round_num, "agent": "con"},
            ) as con_span:
                con_messages = []
                if round_num == 1:
                    user_msg = con_system + "\n\n" + reply_turn_tmpl.format(
                        opponent_stance=labels["pro"],
                        opponent_msg=pro_text,
                        stance=labels["con"],
                    )
                else:
                    user_msg = reply_turn_tmpl.format(
                        opponent_stance=labels["pro"],
                        opponent_msg=pro_text,
                        stance=labels["con"],
                    )
                con_messages.extend(history_con)
                con_messages.append({"role": "user", "content": user_msg})

                with con_span.start_as_current_generation(
                    name=f"llm-con-round-{round_num}",
                    model=LITELLM_MODEL,
                    input=con_messages,
                ) as con_gen:
                    con_response = llm_client.chat.completions.create(
                        model=LITELLM_MODEL,
                        messages=con_messages,
                        max_tokens=512,
                        temperature=0.7,
                    )
                    con_text = con_response.choices[0].message.content
                    con_gen.update(
                        output=con_text,
                        usage_details={
                            "input": con_response.usage.prompt_tokens,
                            "output": con_response.usage.completion_tokens,
                        },
                    )

                con_span.update(output={"message": con_text})
                turns.append(DebateTurn(
                    round=round_num, agent="con", stance=labels["con"], message=con_text,
                ))
                history_con.append({"role": "user", "content": user_msg})
                history_con.append({"role": "assistant", "content": con_text})

        root_span.update(output={"total_turns": len(turns), "rounds_completed": rounds})
        trace_id = langfuse.get_current_trace_id()

    langfuse.flush()
    return turns, trace_id or ""
