import unittest
from unittest.mock import patch, MagicMock, mock_open
import os
import pickle
import requests
import sys
import time
from bs4 import BeautifulSoup

# Add colors for better visualization
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
ENDC = '\033[0m'

def print_step(message):
    """Helper function to print test steps with nice formatting"""
    print(f"{BLUE}>>> {message}{ENDC}")

def print_success(message):
    """Helper function to print success messages"""
    print(f"{GREEN}✓ {message}{ENDC}")

def print_failure(message):
    """Helper function to print failure messages"""
    print(f"{RED}✗ {message}{ENDC}")

# Import the functions to test
# NOTE: Adjust the import statement to match your actual module name
try:
    from main import (
        fetch_page, 
        extract_text, 
        get_article_links, 
        create_rag_pipeline,
        initialize_mistral_model,
        medical_query_input,
        save_to_file,
        scrape_alphabets
    )
except ImportError:
    print(f"{RED}Error: Could not import from main. Make sure the module exists and is in the Python path.{ENDC}")
    sys.exit(1)

class TestWebScrapingFunctions(unittest.TestCase):
    
    def setUp(self):
        print(f"\n{YELLOW}-- Running: {self._testMethodName} --{ENDC}")
    
    @patch('requests.get')
    def test_fetch_page_success(self, mock_get):
        print_step("Testing successful page fetch")
        
        # Setup
        print_step("Setting up mock response with status code 200")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "<html><body>Test content</body></html>"
        mock_get.return_value = mock_response
        
        # Execute
        print_step("Calling fetch_page() with test URL")
        result = fetch_page("https://example.com")
        
        # Assert
        print_step("Verifying returned content matches expected")
        self.assertEqual(result, "<html><body>Test content</body></html>")
        print_step("Verifying URL was called correctly")
        mock_get.assert_called_once_with("https://example.com")
        print_success("fetch_page() successfully returns content for status code 200")
    
    @patch('requests.get')
    def test_fetch_page_failure(self, mock_get):
        print_step("Testing failed page fetch")
        
        # Setup
        print_step("Setting up mock response with status code 404")
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response
        
        # Execute
        print_step("Calling fetch_page() with invalid URL")
        result = fetch_page("https://example.com/not-found")
        
        # Assert
        print_step("Verifying None is returned for failed request")
        self.assertIsNone(result)
        print_step("Verifying URL was called correctly")
        mock_get.assert_called_once_with("https://example.com/not-found")
        print_success("fetch_page() correctly returns None for failed requests")
    
    def test_extract_text(self):
        print_step("Testing HTML text extraction")
        
        # Setup
        print_step("Creating sample HTML with sections to extract")
        html_content = """
        <html>
            <body>
                <h1 class="with-also" itemprop="name">Test Article</h1>
                <div class="section">
                    <div class="section-title">Section 1</div>
                    <div class="section-body">Content 1</div>
                </div>
                <div class="section">
                    <div class="section-title">Section 2</div>
                    <div class="section-body">Content 2</div>
                </div>
                <div class="section">
                    <div class="section-title">Images</div>
                    <div class="section-body">Should be excluded</div>
                </div>
                <div class="section">
                    <div class="section-title">References</div>
                    <div class="section-body">Should also be excluded</div>
                </div>
            </body>
        </html>
        """
        
        # Execute
        print_step("Calling extract_text() with sample HTML")
        title, content = extract_text(html_content)
        
        # Assert
        print_step("Verifying title extraction")
        self.assertEqual(title, "Test Article")
        print_step("Verifying content includes correct sections")
        self.assertIn("Title: Test Article", content)
        self.assertIn("Section 1", content)
        self.assertIn("Content 1", content)
        self.assertIn("Section 2", content)
        self.assertIn("Content 2", content)
        print_step("Verifying excluded sections are actually excluded")
        self.assertNotIn("Images", content)
        self.assertNotIn("Should be excluded", content)
        self.assertNotIn("References", content)
        print_success("extract_text() correctly extracts title and relevant content while excluding unwanted sections")
    
    @patch('main.fetch_page')
    def test_get_article_links(self, mock_fetch_page):
        print_step("Testing article link extraction")
        
        # Setup
        print_step("Creating mock HTML with article links")
        html_content = """
        <div id="mplus-content">
            <li><a href="article/test1.htm">Test 1</a></li>
            <li><a href="article/test2.htm">Test 2</a></li>
            <li class="special"><a href="article/excluded.htm">Excluded</a></li>
            <li><a href="other/not-article.htm">Not an article</a></li>
        </div>
        """
        mock_fetch_page.return_value = html_content
        base_url = "https://medlineplus.gov/ency/"
        
        # Execute
        print_step("Calling get_article_links() for alphabet 'A'")
        result = get_article_links("A")
        
        # Assert
        print_step("Verifying correct number of links extracted")
        self.assertEqual(len(result), 2)
        print_step("Verifying first link is correct")
        self.assertEqual(result[0], f"{base_url}article/test1.htm")
        print_step("Verifying second link is correct")
        self.assertEqual(result[1], f"{base_url}article/test2.htm")
        print_success("get_article_links() correctly extracts only valid article links")

    @patch('main.fetch_page')
    @patch('main.extract_text')
    @patch('main.save_to_file')
    def test_scrape_alphabets(self, mock_save, mock_extract, mock_fetch):
        print_step("Testing article scraping workflow")
        
        # Setup
        print_step("Setting up mocks for fetch_page, extract_text and save_to_file")
        mock_fetch.side_effect = [
            # First call returns HTML with links
            """<div id="mplus-content">
                <li><a href="article/test1.htm">Test 1</a></li>
            </div>""",
            # Second call returns article content
            "<html><body>Article content</body></html>"
        ]
        mock_extract.return_value = ("Test Title", "Test content with sections")
        
        # Execute
        print_step("Calling scrape_alphabets() with letter 'T'")
        scrape_alphabets(['T'])
        
        # Assert
        print_step("Verifying extract_text was called with article HTML")
        mock_extract.assert_called_once()
        print_step("Verifying save_to_file was called with correct parameters")
        mock_save.assert_called_once_with('T', "Test Title", "Test content with sections")
        print_success("scrape_alphabets() successfully processes and saves article content")


class TestRAGPipelineFunctions(unittest.TestCase):
    
    def setUp(self):
        print(f"\n{YELLOW}-- Running: {self._testMethodName} --{ENDC}")
    
    @patch('main.ChatMistralAI')
    def test_initialize_mistral_model(self, mock_chat_mistral):
        print_step("Testing Mistral model initialization")
        
        # Setup
        print_step("Setting up mock for ChatMistralAI")
        mock_llm = MagicMock()
        mock_chat_mistral.return_value = mock_llm
        
        # Execute
        print_step("Calling initialize_mistral_model()")
        result = initialize_mistral_model()
        
        # Assert
        print_step("Verifying model is returned correctly")
        self.assertEqual(result, mock_llm)
        print_step("Verifying model parameters are correct")
        mock_chat_mistral.assert_called_once_with(
            model="mistral-large-latest",
            temperature=0.2,
            max_retries=2
        )
        print_success("initialize_mistral_model() correctly configures and returns the model")
    
    @patch('os.path.exists')
    @patch('pickle.load')
    @patch('builtins.open', new_callable=mock_open)
    def test_create_rag_pipeline_with_cached_vectorstore(self, mock_file, mock_pickle_load, mock_exists):
        print_step("Testing RAG pipeline creation with cached vectorstore")
        
        # Setup
        print_step("Setting up mocks for file operations and embeddings")
        mock_exists.return_value = True
        mock_vectorstore = MagicMock()
        mock_vectorstore.as_retriever.return_value = MagicMock()
        mock_pickle_load.return_value = mock_vectorstore
        
        # Patch necessary components
        with patch('main.HuggingFaceEmbeddings') as mock_embeddings:
            with patch('main.initialize_mistral_model') as mock_init_model:
                mock_model = MagicMock()
                mock_init_model.return_value = mock_model
                
                with patch('main.ChatPromptTemplate.from_template') as mock_prompt:
                    mock_prompt_template = MagicMock()
                    mock_prompt.return_value = mock_prompt_template
                    
                    # Execute
                    print_step("Calling create_rag_pipeline() with use_cached=True")
                    result = create_rag_pipeline(use_cached=True)
                    
                    # Assert
                    print_step("Verifying vectorstore.pkl existence was checked")
                    mock_exists.assert_called_with("vectorstore.pkl")
                    print_step("Verifying vectorstore.pkl was opened for reading")
                    mock_file.assert_called_with("vectorstore.pkl", "rb")
                    print_step("Verifying pickle.load was called")
                    mock_pickle_load.assert_called_once()
                    print_step("Verifying HuggingFaceEmbeddings was initialized correctly")
                    mock_embeddings.assert_called_once_with(model_name="all-MiniLM-L6-v2")
                    print_step("Verifying Mistral model was initialized")
                    mock_init_model.assert_called_once()
                    print_step("Verifying ChatPromptTemplate was created")
                    mock_prompt.assert_called_once()
                    print_step("Verifying RAG chain was created and returned")
                    self.assertIsNotNone(result)
                    print_success("create_rag_pipeline() successfully created pipeline using cached vectorstore")
    
    @patch('os.path.exists')
    def test_create_rag_pipeline_missing_directory(self, mock_exists):
        print_step("Testing error handling for missing data directory")
        
        # Setup
        print_step("Setting up mock to simulate non-existent directory")
        mock_exists.return_value = False
        
        # Execute & Assert
        print_step("Verifying FileNotFoundError is raised for missing directory")
        with self.assertRaises(FileNotFoundError) as context:
            create_rag_pipeline(data_dir="nonexistent_dir", use_cached=False)
        
        print_step("Verifying error message is correct")
        self.assertIn("does not exist", str(context.exception))
        print_success("create_rag_pipeline() correctly raises FileNotFoundError for missing directories")


class TestQueryFunction(unittest.TestCase):
    
    def setUp(self):
        print(f"\n{YELLOW}-- Running: {self._testMethodName} --{ENDC}")
    
    @patch('main.create_rag_pipeline')
    def test_medical_query_input_success(self, mock_create_pipeline):
        print_step("Testing successful medical query processing")
        
        # Setup
        print_step("Setting up mock chain that returns test response")
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "Test medical response about diabetes"
        mock_create_pipeline.return_value = mock_chain
        
        # Execute
        print_step("Calling medical_query_input() with test query")
        response, context = medical_query_input("What is diabetes?")
        
        # Assert
        print_step("Verifying response matches expected")
        self.assertEqual(response, "Test medical response about diabetes")
        print_step("Verifying context is empty string")
        self.assertEqual(context, "")
        print_step("Verifying pipeline was created with cached=True")
        mock_create_pipeline.assert_called_once_with(use_cached=True)
        print_step("Verifying query was passed to chain.invoke()")
        mock_chain.invoke.assert_called_once_with("What is diabetes?")
        print_success("medical_query_input() successfully processes query and returns response")
    
    @patch('main.create_rag_pipeline')
    def test_medical_query_input_exception(self, mock_create_pipeline):
        print_step("Testing error handling in medical query processing")
        
        # Setup
        print_step("Setting up mock to raise exception")
        mock_create_pipeline.side_effect = Exception("Test error")
        
        # Execute & Assert
        print_step("Verifying exception is handled and re-raised with context")
        with self.assertRaises(Exception) as context:
            medical_query_input("What is diabetes?")
        
        error_msg = str(context.exception)
        print_step(f"Checking error message: '{error_msg}'")
        self.assertIn("Error processing medical query: Test error", error_msg)
        print_success("medical_query_input() correctly handles and reports exceptions")


def run_tests_with_animation():
    """Run tests with a visual loading animation"""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestWebScrapingFunctions)
    suite.addTests(loader.loadTestsFromTestCase(TestRAGPipelineFunctions))
    suite.addTests(loader.loadTestsFromTestCase(TestQueryFunction))
    
    print(f"\n{BLUE}======================================{ENDC}")
    print(f"{BLUE}= MEDICAL RAG SYSTEM - UNIT TESTING ={ENDC}")
    print(f"{BLUE}======================================{ENDC}\n")
    
    print(f"{YELLOW}Test suite prepared with {suite.countTestCases()} test cases{ENDC}")
    time.sleep(1)
    print(f"{YELLOW}Starting tests...{ENDC}\n")
    time.sleep(0.5)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{BLUE}======================================{ENDC}")
    print(f"{BLUE}= TEST RESULTS SUMMARY             ={ENDC}")
    print(f"{BLUE}======================================{ENDC}")
    
    if result.wasSuccessful():
        print(f"{GREEN}✓ All tests passed successfully!{ENDC}")
    else:
        print(f"{RED}✗ Some tests failed!{ENDC}")
    
    print(f"Total tests: {result.testsRun}")
    print(f"Passed: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failed: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"{BLUE}======================================{ENDC}")


if __name__ == '__main__':
    run_tests_with_animation()