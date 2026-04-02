import unittest
import sys
import os
from utils.logging_config import logger

# Ensure the TabuLLM package's root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def run_master_suite():
    """Runs all TabuLLM test files and returns True if all pass, False otherwise."""
    logger.info("="*60)
    logger.info("TabuLLM MASTER TEST SUITE - Global Validation")
    logger.info("="*60)
    
    loader = unittest.TestLoader()
    # Discover all tests in this directory starting with 'test_'
    suite = loader.discover(start_dir=os.path.dirname(__file__), pattern='test_*.py')
    
    # We use verbosity=2 to maintain the detailed output in console
    runner = unittest.TextTestRunner(verbosity=2)
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
