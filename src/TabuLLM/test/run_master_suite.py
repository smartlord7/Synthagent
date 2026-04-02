import unittest
import sys
import os

# Ensure the TabuLLM package's root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.logging_config import logger

class LoggingTestResult(unittest.TextTestResult):
    """Custom TestResult that logs individual test case outcomes via the system logger."""
    
    def addSuccess(self, test):
        super().addSuccess(test)
        logger.info(f"PASS: {test.id()}")
        
    def addError(self, test, err):
        super().addError(test, err)
        logger.error(f"ERROR: {test.id()}")
        logger.error(self._exc_info_to_string(err, test))
        
    def addFailure(self, test, err):
        super().addFailure(test, err)
        logger.error(f"FAILURE: {test.id()}")
        logger.error(self._exc_info_to_string(err, test))

def run_master_suite():
    """Runs all TabuLLM test files and returns True if all pass, False otherwise."""
    logger.info("="*60)
    logger.info("TabuLLM MASTER TEST SUITE - Global Validation")
    logger.info("="*60)
    
    loader = unittest.TestLoader()
    # Discover all tests in this directory starting with 'test_'
    suite = loader.discover(start_dir=os.path.dirname(__file__), pattern='test_*.py')
    
    # We use a custom runner that uses our LoggingTestResult
    runner = unittest.TextTestRunner(verbosity=2, resultclass=LoggingTestResult)
    result = runner.run(suite)
    
    # Check for failures or errors
    if not result.wasSuccessful():
        logger.error("\n" + "!"*60)
        logger.error("CRITICAL: Master Test Suite FAILED.")
        logger.error("="*60)
        return False
        
    logger.info("\n" + "="*60)
    logger.info("SUCCESS: All TabuLLM Functionality Tests Passed.")
    logger.info("="*60)
    return True

if __name__ == "__main__":
    success = run_master_suite()
    sys.exit(0 if success else 1)
