# internal
# external
import pytest
# local
from resources.model import JaNoMiModel

def test_handle_input():
    input_handler = JaNoMiModel()
    user_input = "Hello World"

    # Test if the output of generateOutput matches the expected output
    expected_output = input_handler.prefix + user_input
    return input_handler.generateOutput(user_input) == expected_output

if __name__ == '__main__':
    pytest.main()
