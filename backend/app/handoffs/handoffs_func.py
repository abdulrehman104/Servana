import random
import string
from agents import RunContextWrapper
from context.context import AirlineAgentContext


async def on_seat_booking_handoff(context: RunContextWrapper[AirlineAgentContext]) -> None:
    """Set a random flight number when handed off to the seat booking agent."""
    
    context.context.flight_number = f"FLT-{random.randint(100, 999)}"
    context.context.confirmation_number = "".join(random.choices(string.ascii_uppercase + string.digits, k=6))


async def on_cancellation_handoff(context: RunContextWrapper[AirlineAgentContext]) -> None:
    """Ensure context has a confirmation and flight number when handing off to cancellation."""
    if context.context.confirmation_number is None:
        context.context.confirmation_number = "".join(
            random.choices(string.ascii_uppercase + string.digits, k=6)
        )
    if context.context.flight_number is None:
        context.context.flight_number = f"FLT-{random.randint(100, 999)}"