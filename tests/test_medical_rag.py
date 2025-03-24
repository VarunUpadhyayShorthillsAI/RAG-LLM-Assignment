import unittest
from unittest.mock import patch, MagicMock, mock_open
import os
import pickle
import requests
import sys
from bs4 import BeautifulSoup
from io import StringIO

# Import classes from main module
sys.path.append('.')  # Add current directory to path
from main import (
    Config, WebScraper, DataManager, ScrapingOrchestrator, 
    RAGModelBuilder, MedicalAssistant
)

class TestMedlinePlusScrapingFunctions(unittest.TestCase):
    """Test cases for MedlinePlus-specific web scraping functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.web_scraper = WebScraper(Config.BASE_URL)
        self.data_manager = DataManager()
        self.scraping_orchestrator = ScrapingOrchestrator(self.web_scraper, self.data_manager)
    
    @patch('requests.get')
    def test_fetch_page_success(self, mock_get):
        """Test successful page fetching from MedlinePlus"""
        # Mock a successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "<html><body>MedlinePlus content</body></html>"
        mock_get.return_value = mock_response
        
        # Test with a typical MedlinePlus URL
        medlineplus_url = "https://medlineplus.gov/ency/article/000313.htm"
        result = self.web_scraper.fetch_page(medlineplus_url)
        
        # Assertions
        mock_get.assert_called_once_with(medlineplus_url)
        self.assertEqual(result, "<html><body>MedlinePlus content</body></html>")
        print("Test passed: test_fetch_page_success")
    
    def test_extract_text_medlineplus_format(self):
        """Test extracting text from MedlinePlus HTML format"""
        # Create sample HTML content that mimics MedlinePlus structure
        medlineplus_html = """
        <html>
            <body>
                <h1 class="with-also" itemprop="name">Diabetes</h1>
                <div class="section">
                    <div class="section-title">Definition</div>
                    <div class="section-body">Diabetes is a disease in which your blood glucose, or blood sugar, levels are too high.</div>
                </div>
                <div class="section">
                    <div class="section-title">Causes</div>
                    <div class="section-body">Type 1 diabetes is caused by the immune system destroying the cells that release insulin.</div>
                </div>
                <div class="section">
                    <div class="section-title">Symptoms</div>
                    <div class="section-body">Symptoms include increased thirst, frequent urination, hunger, fatigue, and blurred vision.</div>
                </div>
                <div class="section">
                    <div class="section-title">Images</div>
                    <div class="section-body">Image of diabetic retinopathy</div>
                </div>
                <div class="section">
                    <div class="section-title">References</div>
                    <div class="section-body">American Diabetes Association. Standards of medical care in diabetes.</div>
                </div>
                <div class="section">
                    <div class="section-title">Review Date</div>
                    <div class="section-body">1/1/2023</div>
                </div>
            </body>
        </html>
        """
        
        title, content = self.web_scraper.extract_text(medlineplus_html)
        
        # Assertions specific to MedlinePlus content structure
        self.assertEqual(title, "Diabetes")
        self.assertIn("Title: Diabetes", content)
        self.assertIn("Definition", content)
        self.assertIn("Diabetes is a disease in which your blood glucose", content)
        self.assertIn("Causes", content)
        self.assertIn("Symptoms", content)
        
        # Check that excluded sections are not present
        self.assertNotIn("Images", content)
        self.assertNotIn("References", content)
        self.assertNotIn("Review Date", content)
        print("Test passed: test_extract_text_medlineplus_format")
    
    @patch('main.WebScraper.fetch_page')
    def test_get_article_links_medlineplus(self, mock_fetch_page):
        """Test extracting article links from a MedlinePlus alphabet page"""
        # Mock HTML content that mimics a MedlinePlus encyclopedia page
        mock_html = """
        <div id="mplus-content">
            <ul>
                <li><a href="article/000313.htm">Diabetes</a></li>
                <li><a href="article/001214.htm">Diabetic ketoacidosis</a></li>
                <li><a href="something/else.htm">Not an article</a></li>
                <li class="see"><a href="article/excluded.htm">See Also</a></li>
            </ul>
        </div>
        """
        mock_fetch_page.return_value = mock_html
        
        # Call the function with 'D' for diabetes-related terms
        result = self.web_scraper.get_article_links("D")
        
        # Assertions
        expected_links = [
            f"{Config.BASE_URL}article/000313.htm",
            f"{Config.BASE_URL}article/001214.htm"
        ]
        self.assertEqual(result, expected_links)
        mock_fetch_page.assert_called_once_with(f"{Config.BASE_URL}encyclopedia_D.htm")
        print("Test passed: test_get_article_links_medlineplus")
    
    @patch('main.WebScraper.get_article_links')
    @patch('main.WebScraper.fetch_page')
    @patch('main.WebScraper.extract_text')
    @patch('main.DataManager.save_to_file')
    def test_scrape_alphabets_for_diabetes(self, mock_save, mock_extract, mock_fetch, mock_get_links):
        """Test scraping diabetes-related articles"""
        # Mock the dependencies with realistic MedlinePlus content
        mock_get_links.return_value = [
            f"{Config.BASE_URL}article/000313.htm"  # Diabetes article
        ]
        
        diabetes_html = """
        <html>
            <body>
                <h1 class="with-also" itemprop="name">Diabetes</h1>
                <div class="section">
                    <div class="section-title">Definition</div>
                    <div class="section-body">Diabetes is a disease in which your blood glucose levels are too high.</div>
                </div>
            </body>
        </html>
        """
        mock_fetch.return_value = diabetes_html
        mock_extract.return_value = ("Diabetes", "Title: Diabetes\n\nDefinition\nDiabetes is a disease in which your blood glucose levels are too high.")
        
        # Call the function just for the 'D' alphabet
        result = self.scraping_orchestrator.scrape_alphabets(["D"])
        
        # Assertions
        mock_get_links.assert_called_once_with("D")
        mock_fetch.assert_called_once_with(f"{Config.BASE_URL}article/000313.htm")
        mock_extract.assert_called_once_with(diabetes_html)
        mock_save.assert_called_once_with("D", "Diabetes", "Title: Diabetes\n\nDefinition\nDiabetes is a disease in which your blood glucose levels are too high.")
        print("Test passed: test_scrape_alphabets_for_diabetes")
    
    @patch('os.makedirs')
    @patch('builtins.open', new_callable=mock_open)
    def test_save_to_file_medlineplus_content(self, mock_file, mock_makedirs):
        """Test saving MedlinePlus content to appropriate directory structure"""
        # Call the function with realistic MedlinePlus article data
        self.data_manager.save_to_file("D", "Diabetes", "Title: Diabetes\n\nDefinition\nDiabetes is a disease in which your blood glucose levels are too high.")
        
        # Assertions
        mock_makedirs.assert_called_once_with(os.path.join("articles", "D"), exist_ok=True)
        mock_file.assert_called_once_with(os.path.join("articles", "D", "Diabetes.txt"), "w", encoding="utf-8")
        mock_file().write.assert_called_once_with("Title: Diabetes\n\nDefinition\nDiabetes is a disease in which your blood glucose levels are too high.")
        print("Test passed: test_save_to_file_medlineplus_content")
    
    @patch('requests.get')
    def test_fetch_page_error_handling(self, mock_get):
        """Test error handling in fetch_page for HTTP error responses"""
        # Mock an error response
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response
        
        # Test with a typical MedlinePlus URL
        result = self.web_scraper.fetch_page("https://medlineplus.gov/ency/nonexistent.htm")
        
        # Assertions
        self.assertIsNone(result)
        print("Test passed: test_fetch_page_error_handling")


class TestRAGPipelineFunctions(unittest.TestCase):
    """Test cases for RAG pipeline functions with medical data"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.data_manager = DataManager()
        self.rag_builder = RAGModelBuilder(self.data_manager)
        self.medical_assistant = MedicalAssistant(self.rag_builder)
    
    @patch('main.ChatMistralAI')
    def test_initialize_mistral_model(self, mock_mistral):
        """Test initializing the Mistral model for medical queries"""
        # Mock the Mistral initialization
        mock_instance = MagicMock()
        mock_mistral.return_value = mock_instance
        
        # Call the function
        result = self.rag_builder.initialize_mistral_model()
        
        # Assertions
        mock_mistral.assert_called_once_with(
            model="mistral-large-latest",
            temperature=0.2,  # Low temperature for factual medical responses
            max_retries=2
        )
        self.assertEqual(result, mock_instance)
        print("Test passed: test_initialize_mistral_model")
    
    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open)
    @patch('pickle.load')
    @patch('main.HuggingFaceEmbeddings')
    def test_create_rag_pipeline_cached_medical_data(self, mock_embeddings, 
                                                    mock_pickle_load, mock_open_file, 
                                                    mock_exists):
        """Test creating RAG pipeline with cached medical vectorstore"""
        # Mock dependencies for cached vectorstore scenario
        mock_exists.return_value = True
        mock_vectorstore = MagicMock()
        mock_pickle_load.return_value = mock_vectorstore
        mock_retriever = MagicMock()
        mock_vectorstore.as_retriever.return_value = mock_retriever
        mock_embeddings_instance = MagicMock()
        mock_embeddings.return_value = mock_embeddings_instance
        
        # Call the function
        result = self.rag_builder.build_rag_pipeline(force_rebuild=False)
        
        # Assertions
        mock_exists.assert_called_once_with(Config.VECTORSTORE_PATH)
        mock_open_file.assert_called_once_with(Config.VECTORSTORE_PATH, "rb")
        mock_pickle_load.assert_called_once()
        
        # Verify that a runnable was returned
        self.assertIsNotNone(result)
        print("Test passed: test_create_rag_pipeline_cached_medical_data")
    
    @patch('os.path.exists')
    @patch('main.DataManager.get_document_paths')
    @patch('main.TextLoader')
    @patch('main.RecursiveCharacterTextSplitter.split_documents')
    @patch('main.HuggingFaceEmbeddings')
    @patch('main.FAISS.from_documents')
    @patch('builtins.open', new_callable=mock_open)
    def test_create_vectorstore_from_scratch(self, mock_open, mock_faiss, mock_embeddings, 
                                           mock_split, mock_loader, mock_get_paths, mock_exists):
        """Test creating vectorstore from scratch"""
        # Mock dependencies
        mock_exists.return_value = False
        mock_get_paths.return_value = ["path/to/diabetes.txt", "path/to/hypertension.txt"]
        
        mock_loader_instance = MagicMock()
        mock_loader.return_value = mock_loader_instance
        mock_loader_instance.load.return_value = ["document1", "document2"]
        
        mock_split.return_value = ["chunk1", "chunk2", "chunk3"]
        
        mock_embeddings_instance = MagicMock()
        mock_embeddings.return_value = mock_embeddings_instance
        
        mock_vectorstore = MagicMock()
        mock_faiss.return_value = mock_vectorstore
        
        # Call the function
        result = self.rag_builder.create_vectorstore(force_rebuild=True)
        
        # Assertions
        mock_get_paths.assert_called_once()
        self.assertEqual(mock_loader.call_count, 2)
        mock_split.assert_called_once()
        mock_embeddings.assert_called_once_with(model_name=Config.EMBEDDING_MODEL_NAME)
        mock_faiss.assert_called_once_with(["chunk1", "chunk2", "chunk3"], mock_embeddings_instance)
        self.assertEqual(result, mock_vectorstore)
        print("Test passed: test_create_vectorstore_from_scratch")
    
    @patch('os.environ.get')
    @patch('main.ChatMistralAI')
    def test_initialize_mistral_model_missing_api_key(self, mock_mistral, mock_environ_get):
        """Test handling missing API key when initializing Mistral model"""
        # Mock missing API key
        mock_environ_get.return_value = None
        
        # Assert that the function raises the correct exception
        with self.assertRaises(ValueError) as context:
            self.rag_builder.initialize_mistral_model()
        
        self.assertIn("MISTRAL_API_KEY not found", str(context.exception))
        print("Test passed: test_initialize_mistral_model_missing_api_key")
    
    @patch('main.RAGModelBuilder.build_rag_pipeline')
    def test_medical_assistant_error_handling(self, mock_build_rag):
        """Test error handling in medical assistant"""
        # Create a mock chain that raises an exception
        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = Exception("API error")
        mock_build_rag.return_value = mock_chain
        
        # Patch the rag_chain property
        with patch.object(self.medical_assistant, 'rag_chain', mock_chain):
            # Assert that the function raises the correct exception
            with self.assertRaises(Exception) as context:
                self.medical_assistant.process_query("What is diabetes?")
            
            self.assertIn("Error processing medical query", str(context.exception))
            print("Test passed: test_medical_assistant_error_handling")


class TestExceptionHandling(unittest.TestCase):
    """Test cases for exception handling with MedlinePlus-specific scenarios"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.web_scraper = WebScraper(Config.BASE_URL)
        self.data_manager = DataManager()
        self.rag_builder = RAGModelBuilder(self.data_manager)
    
    @patch('os.path.exists')
    def test_rag_pipeline_missing_directory(self, mock_exists):
        """Test exception when the articles directory doesn't exist"""
        # Mock directory not existing
        mock_exists.side_effect = lambda path: False if path == "nonexistent_dir" else True
        
        # Create a data manager with nonexistent directory
        data_manager = DataManager("nonexistent_dir")
        rag_builder = RAGModelBuilder(data_manager)
        
        # Assert that the function raises the correct exception
        with self.assertRaises(FileNotFoundError) as context:
            rag_builder.create_vectorstore(force_rebuild=True)
        
        self.assertIn("does not exist", str(context.exception))
        print("Test passed: test_rag_pipeline_missing_directory")
    
    @patch('main.WebScraper.fetch_page')
    def test_get_article_links_empty_page(self, mock_fetch_page):
        """Test handling of empty or malformed MedlinePlus pages"""
        # Mock an empty HTML response
        mock_fetch_page.return_value = "<html><body></body></html>"
        
        # Call the function
        result = self.web_scraper.get_article_links("Z")  # Using 'Z' which might have fewer articles
        
        # Assertions - should return empty list, not error
        self.assertEqual(result, [])
        print("Test passed: test_get_article_links_empty_page")
    
    @patch('main.WebScraper.fetch_page')
    def test_extract_text_invalid_html(self, mock_fetch_page):
        """Test extracting text from invalid HTML format"""
        # Test with invalid HTML
        invalid_html = "<html><body>No proper structure here</body></html>"
        
        # Extract text should still return something and not crash
        title, content = self.web_scraper.extract_text(invalid_html)
        
        # Assertions
        self.assertEqual(title, "Untitled")  # Default title when none found
        self.assertIn("Title: Untitled", content)  # Default content
        print("Test passed: test_extract_text_invalid_html")
    
    @patch('main.Config.load_environment')
    def test_config_missing_api_key(self, mock_load_env):
        """Test handling missing API key in config"""
        # Mock missing API key
        mock_load_env.return_value = False
        
        # Call the function and redirect stdout to capture output
        saved_stdout = sys.stdout
        try:
            out = StringIO()
            sys.stdout = out
            
            # Import main module with mocked environment
            from main import main
            with self.assertRaises(SystemExit):
                main()
                
            output = out.getvalue()
            self.assertIn("WARNING: MISTRAL_API_KEY not found", output)
        finally:
            sys.stdout = saved_stdout
        
        print("Test passed: test_config_missing_api_key")


class TestIntegration(unittest.TestCase):
    """Integration tests for MedlinePlus scraper and medical RAG system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.web_scraper = WebScraper(Config.BASE_URL)
        self.data_manager = DataManager()
        self.scraping_orchestrator = ScrapingOrchestrator(self.web_scraper, self.data_manager)
        self.rag_builder = RAGModelBuilder(self.data_manager)
        self.medical_assistant = MedicalAssistant(self.rag_builder)
    
    @patch('main.WebScraper.get_article_links')
    @patch('main.WebScraper.fetch_page')
    @patch('main.WebScraper.extract_text')
    @patch('main.DataManager.save_to_file')
    def test_end_to_end_scraping_workflow(self, mock_save, mock_extract, mock_fetch, mock_get_links):
        """Test the complete MedlinePlus scraping workflow for multiple articles"""
        # Set up mock behavior for multiple articles
        mock_get_links.side_effect = [
            [f"{Config.BASE_URL}article/000313.htm"],  # For D (Diabetes)
            [f"{Config.BASE_URL}article/000468.htm"]   # For H (Hypertension)
        ]
        
        # Mock responses for each article
        mock_html_responses = {
            f"{Config.BASE_URL}article/000313.htm": "<html><body><h1 class='with-also' itemprop='name'>Diabetes</h1></body></html>",
            f"{Config.BASE_URL}article/000468.htm": "<html><body><h1 class='with-also' itemprop='name'>Hypertension</h1></body></html>"
        }
        
        def mock_fetch_side_effect(url):
            return mock_html_responses.get(url, "")
        
        mock_fetch.side_effect = mock_fetch_side_effect
        
        # Mock extract text responses
        mock_extract_responses = {
            "<html><body><h1 class='with-also' itemprop='name'>Diabetes</h1></body></html>": ("Diabetes", "Content about diabetes"),
            "<html><body><h1 class='with-also' itemprop='name'>Hypertension</h1></body></html>": ("Hypertension", "Content about hypertension")
        }
        
        def mock_extract_side_effect(html):
            return mock_extract_responses.get(html, ("Unknown", "Unknown content"))
        
        mock_extract.side_effect = mock_extract_side_effect
        
        # Call the scrape function
        results = self.scraping_orchestrator.scrape_alphabets(["D", "H"])
        
        # Assertions
        self.assertEqual(mock_get_links.call_count, 2)
        self.assertEqual(mock_fetch.call_count, 2)
        self.assertEqual(mock_extract.call_count, 2)
        self.assertEqual(mock_save.call_count, 2)
        
        # Verify specific calls
        mock_save.assert_any_call("D", "Diabetes", "Content about diabetes")
        mock_save.assert_any_call("H", "Hypertension", "Content about hypertension")
        print("Test passed: test_end_to_end_scraping_workflow")
    
    @patch('os.path.exists')
    @patch('main.DataManager.get_document_paths')
    @patch('main.UserInterface.run_menu')
    def test_main_function_initialization(self, mock_run_menu, mock_get_paths, mock_exists):
        """Test main function initialization of all components"""
        # Mock dependencies
        mock_exists.return_value = True
        mock_get_paths.return_value = ["path/to/document.txt"]
        
        # Call main function with redirected stdout
        saved_stdout = sys.stdout
        try:
            out = StringIO()
            sys.stdout = out
            
            # Import main module with mocked functions
            with patch('main.Config.load_environment', return_value=True):
                from main import main
                main()
            
        finally:
            sys.stdout = saved_stdout
        
        # Assert that run_menu was called
        mock_run_menu.assert_called_once()
        print("Test passed: test_main_function_initialization")
    
    @patch('main.WebScraper.get_article_links')
    def test_scrape_alphabet_empty_results(self, mock_get_links):
        """Test scraping an alphabet with no results"""
        # Mock empty results
        mock_get_links.return_value = []
        
        # Call the function
        count = self.scraping_orchestrator.scrape_alphabet("Z")  # Z might have fewer articles
        
        # Assertions
        self.assertEqual(count, 0)
        mock_get_links.assert_called_once_with("Z")
        print("Test passed: test_scrape_alphabet_empty_results")
    
    @patch('builtins.input', side_effect=["1", "A", "4"])
    @patch('main.ScrapingOrchestrator.scrape_alphabet')
    def test_user_interface_scrape_option(self, mock_scrape, mock_input):
        """Test user interface scrape option flow"""
        # Mock scraping result
        mock_scrape.return_value = 10  # 10 articles scraped
        
        # Create a UI instance with mocked components
        ui = UserInterface(self.scraping_orchestrator, self.rag_builder, self.medical_assistant)
        
        # Redirect stdout to capture output
        saved_stdout = sys.stdout
        try:
            out = StringIO()
            sys.stdout = out
            
            # Run menu
            ui.run_menu()
            
            output = out.getvalue()
            
            # Assertions
            mock_scrape.assert_called_once_with("A")
            self.assertIn("10 articles scraped for alphabet 'A'", output)
            self.assertIn("Scraping completed successfully!", output)
            
        finally:
            sys.stdout = saved_stdout
        
        print("Test passed: test_user_interface_scrape_option")


# Custom TextTestRunner that logs each test as it runs
class LoggingTestRunner(unittest.TextTestRunner):
    def run(self, test):
        # Run the test
        print("\nRunning tests line by line:")
        print("="*50)
        result = super().run(test)
        return result

# Main test runner
if __name__ == '__main__':
    # Create a test suite to run
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes to the suite
    suite.addTests(loader.loadTestsFromTestCase(TestMedlinePlusScrapingFunctions))
    suite.addTests(loader.loadTestsFromTestCase(TestRAGPipelineFunctions))
    suite.addTests(loader.loadTestsFromTestCase(TestExceptionHandling))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run the tests with our custom runner
    result = LoggingTestRunner(verbosity=2).run(suite)
    
    # Print test summary without emojis
    print("\n" + "="*50)
    print("MEDLINEPLUS SCRAPER TEST SUMMARY")
    print("="*50)
    
    total_tests = result.testsRun
    passed_tests = total_tests - len(result.errors) - len(result.failures)
    
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    
    if result.errors or result.failures:
        print(f"Failed: {len(result.errors) + len(result.failures)}")
        
        if result.errors:
            print("\nErrors:")
            for test, error in result.errors:
                print(f"  - {test}")
                print(f"    {error.splitlines()[0]}")
        
        if result.failures:
            print("\nFailures:")
            for test, failure in result.failures:
                print(f"  - {test}")
                print(f"    {failure.splitlines()[0]}")
    else:
        print("All tests passed!")
    
    print("="*50)
    
    # Exit with appropriate code
    sys.exit(not result.wasSuccessful())