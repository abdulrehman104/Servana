from context.context import AirlineAgentContext
from agents import function_tool, RunContextWrapper


@function_tool(name_override="faq_lookup_tool", description_override="Lookup frequently asked questions.")
async def faq_lookup_tool(question: str) -> str:
    """Lookup answers to frequently asked questions."""
    
    q = question.lower()
    if "bag" in q or "baggage" in q:
        return (
            "You are allowed to bring one bag on the plane. "
            "It must be under 50 pounds and 22 inches x 14 inches x 9 inches."
        )
    
    elif "seats" in q or "plane" in q:
        return (
            "There are 120 seats on the plane. "
            "There are 22 business class seats and 98 economy seats. "
            "Exit rows are rows 4 and 16. "
            "Rows 5-8 are Economy Plus, with extra legroom."
        )
    
    elif "wifi" in q:
        return "We have free wifi on the plane, join Airline-Wifi"
    return "I'm sorry, I don't know the answer to that question."


@function_tool
async def update_seat(context: RunContextWrapper[AirlineAgentContext], confirmation_number: str, new_seat: str) -> str:
    """Update the seat for a given confirmation number."""

    context.context.confirmation_number = confirmation_number
    context.context.seat_number = new_seat
    assert context.context.flight_number is not None, "Flight number is required"
    return f"Updated seat to {new_seat} for confirmation number {confirmation_number}"


@function_tool(name_override="flight_status_tool", description_override="Lookup status for a flight.")
async def flight_status_tool(flight_number: str) -> str:
    """Lookup the status for a flight."""
    return f"Flight {flight_number} is on time and scheduled to depart at gate A10."


@function_tool(name_override="baggage_tool", description_override="Lookup baggage allowance and fees.")
async def baggage_tool(query: str) -> str:
    """Lookup baggage allowance and fees."""

    q = query.lower()
    if "fee" in q:
        return "Overweight bag fee is $75."
    if "allowance" in q:
        return "One carry-on and one checked bag (up to 50 lbs) are included."
    return "Please provide details about your baggage inquiry."


@function_tool(name_override="display_seat_map", description_override="Display an interactive seat map to the customer so they can choose a new seat.")
async def display_seat_map(context: RunContextWrapper[AirlineAgentContext]) -> str:
    """Trigger the UI to show an interactive seat map to the customer."""
    # The returned string will be interpreted by the UI to open the seat selector.
    return "DISPLAY_SEAT_MAP"

@function_tool(name_override="cancel_flight", description_override="Cancel a flight.")
async def cancel_flight(context: RunContextWrapper[AirlineAgentContext]) -> str:
    """Cancel the flight in the context."""
    
    fn = context.context.flight_number
    assert fn is not None, "Flight number is required"
    return f"Flight {fn} successfully cancelled"