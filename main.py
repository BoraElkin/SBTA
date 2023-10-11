import interface.text_ui as ui

def main():
    while True:
        # Get user input
        user_input = ui.get_user_input()
        
        # Check for exit commands
        if user_input.lower() in ["exit", "quit"]:
            ui.display_exit_message()
            break

        # Generate a response (dummy function for now)
        response = generate_response(user_input)

        # Display the response using the text_ui module
        ui.display_message(response)


def generate_response(user_input):
    # This is a dummy function, just echoing back the input for now.
    return f"You said: {user_input}"


if __name__ == "__main__":
    main()
