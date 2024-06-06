import unittest
from chatbot import (
    setup_chatbot, get_response_from_openai, read_pdf, preprocess_input,
    find_most_similar_document, get_document_response
)

class TestChatbotFunctions(unittest.TestCase):
    def setUp(self):
        setup_chatbot()
    
    def test_get_response_from_openai(self):
        response = get_response_from_openai("Hello")
        self.assertIsInstance(response, str)

    def test_read_pdf(self):
        # Create a temporary PDF file for testing
        import tempfile
        from fpdf import FPDF
        
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="This is a test PDF file.", ln=True, align='C')
        
        temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        pdf.output(temp_pdf.name)
        
        text = read_pdf(temp_pdf.name)
        self.assertIn("This is a test PDF file.", text)
        
        temp_pdf.close()
        os.remove(temp_pdf.name)
    
    def test_preprocess_input(self):
        user_input = "Hello, how are you?"
        processed_input = preprocess_input(user_input)
        self.assertEqual(processed_input, "hello ?")
    
    def test_find_most_similar_document(self):
        documents = ["This is a test document.", "Another test document here."]
        user_input = "test document"
        index = find_most_similar_document(user_input, documents)
        self.assertEqual(index, 0)
    
    def test_get_document_response(self):
        documents = ["This is a test document.", "Another test document here."]
        user_input = "test document"
        response = get_document_response(user_input, documents)
        self.assertEqual(response, "This is a test document.")

if __name__ == '__main__':
    unittest.main()
