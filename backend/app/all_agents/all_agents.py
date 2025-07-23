import os 
from dotenv import load_dotenv
from context.context import AirlineAgentContext
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX
from guadrails.guadrails import relevance_guardrail, jailbreak_guardrail
from handoffs.handoffs_func import on_cancellation_handoff, on_seat_booking_handoff
from tools.tools import update_seat, display_seat_map, flight_status_tool, faq_lookup_tool, cancel_flight
from agents import Agent, function_tool, handoff, RunContextWrapper, AsyncOpenAI, OpenAIChatCompletionsModel, set_tracing_disabled


# ————————— Api & LLM Config —————————————————————————
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set.")

CLIENT = AsyncOpenAI(api_key=GEMINI_API_KEY, base_url="https://generativelanguage.googleapis.com")
set_tracing_disabled(disabled=True)

# ————————— Agent Instructions —————————————————————————
def seat_booking_instructions(run_context: RunContextWrapper[AirlineAgentContext], agent: Agent[AirlineAgentContext]) -> str:
    ctx = run_context.context
    confirmation = ctx.confirmation_number or "[unknown]"
    return (
        f"{RECOMMENDED_PROMPT_PREFIX}\n"
        "You are a seat booking agent. If you are speaking to a customer, you probably were transferred to from the triage agent.\n"
        "Use the following routine to support the customer.\n"
        f"1. The customer's confirmation number is {confirmation}."+
        "If this is not available, ask the customer for their confirmation number. If you have it, confirm that is the confirmation number they are referencing.\n"
        "2. Ask the customer what their desired seat number is. You can also use the display_seat_map tool to show them an interactive seat map where they can click to select their preferred seat.\n"
        "3. Use the update seat tool to update the seat on the flight.\n"
        "If the customer asks a question that is not related to the routine, transfer back to the triage agent."
    )

def flight_status_instructions(run_context: RunContextWrapper[AirlineAgentContext], agent: Agent[AirlineAgentContext]) -> str:
    ctx = run_context.context
    confirmation = ctx.confirmation_number or "[unknown]"
    flight = ctx.flight_number or "[unknown]"
    return (
        f"{RECOMMENDED_PROMPT_PREFIX}\n"
        "You are a Flight Status Agent. Use the following routine to support the customer:\n"
        f"1. The customer's confirmation number is {confirmation} and flight number is {flight}.\n"
        "   If either is not available, ask the customer for the missing information. If you have both, confirm with the customer that these are correct.\n"
        "2. Use the flight_status_tool to report the status of the flight.\n"
        "If the customer asks a question that is not related to flight status, transfer back to the triage agent."
    )

def cancellation_instructions(run_context: RunContextWrapper[AirlineAgentContext], agent: Agent[AirlineAgentContext]) -> str:
    ctx = run_context.context
    confirmation = ctx.confirmation_number or "[unknown]"
    flight = ctx.flight_number or "[unknown]"
    return (
        f"{RECOMMENDED_PROMPT_PREFIX}\n"
        "You are a Cancellation Agent. Use the following routine to support the customer:\n"
        f"1. The customer's confirmation number is {confirmation} and flight number is {flight}.\n"
        "   If either is not available, ask the customer for the missing information. If you have both, confirm with the customer that these are correct.\n"
        "2. If the customer confirms, use the cancel_flight tool to cancel their flight.\n"
        "If the customer asks anything else, transfer back to the triage agent."
    )


# ————————— Define Agents —————————————————————————
seat_booking_agent = Agent[AirlineAgentContext](
    name="Seat Booking Agent",
    instructions=seat_booking_instructions,
    model=OpenAIChatCompletionsModel(model="gemini-1.5-flash", openai_client=CLIENT),
    tools=[update_seat, display_seat_map],
    handoff_description="A helpful agent that can update a seat on a flight.",
    input_guardrails=[relevance_guardrail, jailbreak_guardrail],
)

flight_status_agent = Agent[AirlineAgentContext](
    name="Flight Status Agent",
    instructions=flight_status_instructions,
    model=OpenAIChatCompletionsModel(model="gemini-1.5-pro", openai_client=CLIENT),
    tools=[flight_status_tool],
    handoff_description="An agent to provide flight status information.",
    input_guardrails=[relevance_guardrail, jailbreak_guardrail],
)

cancellation_agent = Agent[AirlineAgentContext](
    name="Cancellation Agent",
    instructions=cancellation_instructions,
    model=OpenAIChatCompletionsModel(model="gemini-2.0-flash", openai_client=CLIENT),
    tools=[cancel_flight],
    handoff_description="An agent to cancel flights.",   
    input_guardrails=[relevance_guardrail, jailbreak_guardrail],
)

faq_agent = Agent[AirlineAgentContext](
    name="FAQ Agent",
    instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
    You are an FAQ agent. If you are speaking to a customer, you probably were transferred to from the triage agent.
    Use the following routine to support the customer.
    1. Identify the last question asked by the customer.
    2. Use the faq lookup tool to get the answer. Do not rely on your own knowledge.
    3. Respond to the customer with the answer""",
    model=OpenAIChatCompletionsModel(model="gemini-2.0-flash", openai_client=CLIENT),
    tools=[faq_lookup_tool],
    handoff_description="A helpful agent that can answer questions about the airline.",
    input_guardrails=[relevance_guardrail, jailbreak_guardrail],
)


