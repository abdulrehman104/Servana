import os
import asyncio
from dotenv import load_dotenv
from context.context import AirlineAgentContext
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX
from guadrails.guadrails import relevance_guardrail, jailbreak_guardrail
from handoffs.handoffs_func import on_cancellation_handoff, on_seat_booking_handoff
from all_agents.all_agents import flight_status_agent, cancellation_agent, faq_agent, seat_booking_agent
from agents import Agent, handoff, AsyncOpenAI, OpenAIChatCompletionsModel, set_tracing_disabled, Runner


# ————————— Api & LLM Config —————————————————————————
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set.")

CLIENT = AsyncOpenAI(api_key=GEMINI_API_KEY, base_url="https://generativelanguage.googleapis.com")
set_tracing_disabled(disabled=True)


# ————————— Main Function —————————————————————————
async def main(prompt: str):
    # ————————— Triage Agent —————————————————————————
    triage_agent = Agent[AirlineAgentContext](
        name="Triage Agent",
        instructions=(
            f"{RECOMMENDED_PROMPT_PREFIX} "
            "You are a helpful triaging agent. You can use your tools to delegate questions to other appropriate agents."
        ),
        model=OpenAIChatCompletionsModel(model="gemini-2.0-flash", openai_client=CLIENT),
        handoff_description="A triage agent that can delegate a customer's request to the appropriate agent.",
        handoffs=[
            flight_status_agent,
            handoff(agent=cancellation_agent, on_handoff=on_cancellation_handoff),
            faq_agent,
            handoff(agent=seat_booking_agent, on_handoff=on_seat_booking_handoff),
        ],
        input_guardrails=[relevance_guardrail, jailbreak_guardrail],
    )

    print("Triage Agent initialized with handoffs to other agents.")
    result = await Runner.run(triage_agent, prompt)
    print("Triage Agent run completed.")
    print("Result:", result.final_output)

if __name__ == "__main__":
    # test_prompt = "I need help with my flight booking."
    prompt = "I need to change my seat on a flight."
    asyncio.run(main(prompt))


# # ————————— Setup Handoffs Relationships —————————————————————————
# faq_agent.handoffs.append(triage_agent)
# seat_booking_agent.handoffs.append(triage_agent)
# flight_status_agent.handoffs.append(triage_agent)
# cancellation_agent.handoffs.append(triage_agent)