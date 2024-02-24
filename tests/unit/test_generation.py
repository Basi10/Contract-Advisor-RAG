import unittest
from unittest.mock import patch, Mock
from src.generation import Generation  # Replace 'your_module' with the actual module name containing the Generation class

class TestGeneration(unittest.TestCase):

    @patch("src.generation.OpenAI")
    def setUp(self, mock_openai):
        self.generation = Generation(model_name="gpt-4-turbo-preview")

    def test_chats(self):
        # Test the chats method
        response_content = "Generated response"
        self.generation.chat = Mock(return_value=response_content)

        response = self.generation.chats("Hello, how are you?")

        self.assertEqual(response, response_content)

    def test_get_completion(self):
        # Test the get_completion method
        messages = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Who won the world series in 2020?"}]
        completion_content = "Generated completion"
        self.generation.client.chat.completions.create.return_value = completion_content

        completion = self.generation.get_completion(messages=messages)

        self.assertEqual(completion, completion_content)

if __name__ == '__main__':
    unittest.main()
