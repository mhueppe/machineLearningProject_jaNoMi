import pytest
from resources.inputHandler import InputHandler


# Define a test function to test the InputHandler
def test_handle_input():
    input_handler = InputHandler()
    user_input = "Hello World"

    # Test if the output of handleInput matches the expected output
    expected_output = input_handler.prefix + user_input
    assert input_handler.handleInput(user_input) == expected_output, "Input was not handled properly"


# If you want to run the tests directly (not usually necessary with pytest)
if __name__ == '__main__':
    pytest.main()
