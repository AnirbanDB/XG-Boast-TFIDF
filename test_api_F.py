# Test client for Firco XGBoost API
# Co        print("\nTesting root endpoint...")
        try:
            response = await client.get("/")
            if response.status_code == 200:
                data = response.json()
                print(f"PASS: Root endpoint: {data.get('message', 'OK')}")
                return True
            else:
                print(f"FAIL: Root endpoint failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"FAIL: Root endpoint error: {e}")
            return Falseesting of all API endpoints

import requests
import json
import time
import os
import pandas as pd
from typing import Dict, Any, Optional
import sys
from pathlib import Path

# Configuration
API_BASE_URL = "http://localhost:3004"
API_PREFIX = "/v1/firco-xgb"
TEST_DATA_PATH = "../firco_alerts_final_5000_7.csv"

class FircoXGBoostAPITester:
    """Comprehensive tester for Firco XGBoost API."""
    
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
        
    def _make_request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        """Make HTTP request with error handling."""
        url = f"{self.base_url}{API_PREFIX}{endpoint}"
        try:
            response = self.session.request(method, url, **kwargs)
            return response
        except requests.exceptions.RequestException as e:
            print(f"FAIL: Request failed: {e}")
            raise
    
    def test_root_endpoint(self) -> bool:
        """Test root endpoint."""
        print("\nINFO: Testing root endpoint...")
        try:
            response = self.session.get(f"{self.base_url}/")
            if response.status_code == 200:
                data = response.json()
                print(f"PASS: Root endpoint: {data.get('message', 'OK')}")
                return True
            else:
                print(f"FAIL: Root endpoint failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"FAIL: Root endpoint error: {e}")
            return False
    
    def test_health_check(self) -> bool:
        """Test health check endpoint."""
        print("\nINFO: Testing health check...")
        try:
            response = self._make_request('GET', '/health')
            if response.status_code == 200:
                data = response.json()
                print(f"PASS: Health check: {data.get('status', 'Unknown')}")
                print(f"   Model loaded: {data.get('model_loaded', False)}")
                print(f"   Training status: {data.get('training_status', 'Unknown')}")
                return True
            else:
                print(f"FAIL: Health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"FAIL: Health check error: {e}")
            return False
    
    def test_training_status(self) -> bool:
        """Test training status endpoint."""
        print("\nINFO: Testing training status...")
        try:
            response = self._make_request('GET', '/training-status')
            if response.status_code == 200:
                data = response.json()
                training_status = data.get('training_status', {})
                print(f"PASS: Training status: {training_status.get('status', 'Unknown')}")
                print(f"   Is training: {training_status.get('is_training', False)}")
                print(f"   Message: {training_status.get('message', 'No message')}")
                return True
            else:
                print(f"FAIL: Training status failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"FAIL: Training status error: {e}")
            return False
    
    def test_list_models(self) -> bool:
        """Test list models endpoint."""
        print("\nINFO: Testing list models...")
        try:
            response = self._make_request('GET', '/models')
            if response.status_code == 200:
                data = response.json()
                current_models = data.get('current_models', [])
                archived_models = data.get('archived_models', [])
                print(f"PASS: Models listed successfully")
                print(f"   Current models: {len(current_models)}")
                print(f"   Archived models: {len(archived_models)}")
                print(f"   Total models: {data.get('total_models', 0)}")
                return True
            else:
                print(f"FAIL: List models failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"FAIL: List models error: {e}")
            return False
    
    def test_training(self, data_path: str = TEST_DATA_PATH) -> Optional[str]:
        """Test model training endpoint."""
        print("\nINFO: Testing model training...")
        
        # Check if data file exists
        if not os.path.exists(data_path):
            print(f"FAIL: Test data file not found: {data_path}")
            return None
        
        try:
            # Prepare training configuration
            config = {
                "label_cols": ["hit.review_decision", "hit.review_comments", "decision.last_action", "decision.reviewer_comments"],
                "text_col": "hit.matching_text",
                "random_state": 42,
                "test_size": 0.2
            }
            
            # Prepare files and data
            files = {
                'file': ('firco_test_data.csv', open(data_path, 'rb'), 'text/csv')
            }
            
            data = {
                'config': json.dumps(config)
            }
            
            # Remove Content-Type header for multipart/form-data
            headers = {k: v for k, v in self.session.headers.items() if k.lower() != 'content-type'}
            
            # Make request
            url = f"{self.base_url}{API_PREFIX}/train"
            response = requests.post(url, files=files, data=data, headers=headers)
            
            # Close file
            files['file'][1].close()
            
            if response.status_code == 200:
                result = response.json()
                print(f"PASS: Training started successfully")
                print(f"   Training ID: {result.get('training_id', 'Unknown')}")
                print(f"   Status: {result.get('status', 'Unknown')}")
                print(f"   Estimated time: {result.get('estimated_time', 'Unknown')}")
                return result.get('training_id')
            else:
                print(f"FAIL: Training failed: {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"   Error: {error_data.get('detail', 'Unknown error')}")
                except:
                    print(f"   Error: {response.text}")
                return None
                
        except Exception as e:
            print(f"FAIL: Training error: {e}")
            return None
    
    def wait_for_training_completion(self, timeout: int = 1800) -> bool:
        """Wait for training to complete."""
        print("\n⏳ Waiting for training to complete...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                response = self._make_request('GET', '/training-status')
                if response.status_code == 200:
                    data = response.json()
                    training_status = data.get('training_status', {})
                    status = training_status.get('status', 'unknown')
                    is_training = training_status.get('is_training', False)
                    progress = training_status.get('progress', 0)
                    current_stage = training_status.get('current_stage', 'unknown')
                    
                    print(f"   Status: {status} | Progress: {progress}% | Stage: {current_stage}")
                    
                    if status == 'completed' and not is_training:
                        print("PASS: Training completed successfully!")
                        return True
                    elif status == 'failed':
                        error = training_status.get('error', 'Unknown error')
                        print(f"FAIL: Training failed: {error}")
                        return False
                    
                    time.sleep(10)  # Wait 10 seconds before checking again
                else:
                    print(f"FAIL: Error checking training status: {response.status_code}")
                    return False
                    
            except Exception as e:
                print(f"FAIL: Error waiting for training: {e}")
                return False
        
        print(f"FAIL: Training timeout after {timeout} seconds")
        return False
    
    def test_single_prediction(self, text: str = "High-risk PEP match on organization Sheppard-Johnson with score 0.95") -> bool:
        """Test single text prediction."""
        print("\nINFO: Testing single prediction...")
        
        try:
            # Prepare data
            data = {
                'text': text
            }
            
            # Remove Content-Type header for form data
            headers = {k: v for k, v in self.session.headers.items() if k.lower() != 'content-type'}
            
            # Make request
            url = f"{self.base_url}{API_PREFIX}/predict"
            response = requests.post(url, data=data, headers=headers)
            
            if response.status_code == 200:
                result = response.json()
                predictions = result.get('predictions', [])
                print(f"PASS: Single prediction successful")
                print(f"   Model version: {result.get('model_version', 'Unknown')}")
                print(f"   Processing time: {result.get('processing_time', 0):.3f}s")
                
                if predictions:
                    pred_data = predictions[0]
                    print(f"   Input text: {pred_data.get('text', '')[:100]}...")
                    pred_results = pred_data.get('predictions', {})
                    for target, pred in pred_results.items():
                        predicted_class = pred.get('predicted_class', 'Unknown')
                        print(f"   {target}: {predicted_class}")
                
                return True
            else:
                print(f"FAIL: Single prediction failed: {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"   Error: {error_data.get('detail', 'Unknown error')}")
                except:
                    print(f"   Error: {response.text}")
                return False
                
        except Exception as e:
            print(f"FAIL: Single prediction error: {e}")
            return False
    
    def test_batch_prediction(self, data_path: str = TEST_DATA_PATH) -> bool:
        """Test batch prediction with file upload."""
        print("\nINFO: Testing batch prediction...")
        
        # Check if data file exists
        if not os.path.exists(data_path):
            print(f"FAIL: Test data file not found: {data_path}")
            return False
        
        try:
            # Create a smaller test file (first 10 rows)
            df = pd.read_csv(data_path)
            test_df = df.head(10)
            test_file_path = "test_batch_prediction.csv"
            test_df.to_csv(test_file_path, index=False)
            
            # Prepare file upload
            files = {
                'file': ('test_batch.csv', open(test_file_path, 'rb'), 'text/csv')
            }
            
            # Remove Content-Type header for multipart/form-data
            headers = {k: v for k, v in self.session.headers.items() if k.lower() != 'content-type'}
            
            # Make request
            url = f"{self.base_url}{API_PREFIX}/predict"
            response = requests.post(url, files=files, headers=headers)
            
            # Close file and clean up
            files['file'][1].close()
            if os.path.exists(test_file_path):
                os.remove(test_file_path)
            
            if response.status_code == 200:
                result = response.json()
                predictions = result.get('predictions', [])
                print(f"PASS: Batch prediction successful")
                print(f"   Model version: {result.get('model_version', 'Unknown')}")
                print(f"   Processing time: {result.get('processing_time', 0):.3f}s")
                print(f"   Predictions count: {len(predictions)}")
                
                if predictions:
                    # Show first prediction
                    pred_data = predictions[0]
                    pred_results = pred_data.get('predictions', {})
                    print(f"   Sample prediction:")
                    for target, pred in pred_results.items():
                        predicted_class = pred.get('predicted_class', 'Unknown')
                        print(f"     {target}: {predicted_class}")
                
                return True
            else:
                print(f"FAIL: Batch prediction failed: {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"   Error: {error_data.get('detail', 'Unknown error')}")
                except:
                    print(f"   Error: {response.text}")
                return False
                
        except Exception as e:
            print(f"FAIL: Batch prediction error: {e}")
            # Clean up test file
            if os.path.exists(test_file_path):
                os.remove(test_file_path)
            return False
    
    def test_validation(self, data_path: str = TEST_DATA_PATH) -> bool:
        """Test model validation endpoint."""
        print("\nINFO: Testing model validation...")
        
        # Check if data file exists
        if not os.path.exists(data_path):
            print(f"FAIL: Test data file not found: {data_path}")
            return False
        
        try:
            # Create a smaller validation file
            df = pd.read_csv(data_path)
            val_df = df.head(50)
            val_file_path = "test_validation.csv"
            val_df.to_csv(val_file_path, index=False)
            
            # Prepare file upload
            files = {
                'file': ('validation.csv', open(val_file_path, 'rb'), 'text/csv')
            }
            
            # Remove Content-Type header for multipart/form-data
            headers = {k: v for k, v in self.session.headers.items() if k.lower() != 'content-type'}
            
            # Make request
            url = f"{self.base_url}{API_PREFIX}/validate"
            response = requests.post(url, files=files, headers=headers)
            
            # Close file and clean up
            files['file'][1].close()
            if os.path.exists(val_file_path):
                os.remove(val_file_path)
            
            if response.status_code == 200:
                result = response.json()
                print(f"PASS: Model validation successful")
                print(f"   Model version: {result.get('model_version', 'Unknown')}")
                print(f"   Message: {result.get('message', 'No message')}")
                
                validation_results = result.get('validation_results', {})
                if validation_results:
                    print(f"   Validation results available")
                
                return True
            else:
                print(f"FAIL: Model validation failed: {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"   Error: {error_data.get('detail', 'Unknown error')}")
                except:
                    print(f"   Error: {response.text}")
                return False
                
        except Exception as e:
            print(f"FAIL: Model validation error: {e}")
            # Clean up test file
            if os.path.exists(val_file_path):
                os.remove(val_file_path)
            return False
    
    def test_feature_importance(self, model_version: str = "firco_xgb_model.pkl") -> bool:
        """Test feature importance endpoint."""
        print("\nINFO: Testing feature importance...")
        
        try:
            response = self._make_request('GET', f'/feature-importance/{model_version}')
            
            if response.status_code == 200:
                result = response.json()
                print(f"PASS: Feature importance retrieved successfully")
                print(f"   Model version: {result.get('model_version', 'Unknown')}")
                
                feature_importance = result.get('feature_importance', {})
                if feature_importance:
                    print(f"   Feature importance data available for {len(feature_importance)} targets")
                    for target, importance_data in feature_importance.items():
                        print(f"     {target}: {len(importance_data)} features")
                
                return True
            else:
                print(f"FAIL: Feature importance failed: {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"   Error: {error_data.get('detail', 'Unknown error')}")
                except:
                    print(f"   Error: {response.text}")
                return False
                
        except Exception as e:
            print(f"FAIL: Feature importance error: {e}")
            return False
    
    def test_performance_report(self, model_version: str = "firco_xgb_model.pkl") -> bool:
        """Test performance report endpoint."""
        print("\nINFO: Testing performance report...")
        
        try:
            response = self._make_request('GET', f'/performance-report/{model_version}')
            
            if response.status_code == 200:
                result = response.json()
                print(f"PASS: Performance report retrieved successfully")
                print(f"   Model version: {result.get('model_version', 'Unknown')}")
                
                performance_data = result.get('performance_data', {})
                if performance_data:
                    print(f"   Performance data available")
                
                return True
            else:
                print(f"FAIL: Performance report failed: {response.status_code}")
                try:
                    error_data = response.json()
                    print(f"   Error: {error_data.get('detail', 'Unknown error')}")
                except:
                    print(f"   Error: {response.text}")
                return False
                
        except Exception as e:
            print(f"FAIL: Performance report error: {e}")
            return False
    
    def run_comprehensive_test(self) -> bool:
        """Run all API tests in sequence."""
        print("=" * 80)
        print("STARTING: STARTING COMPREHENSIVE FIRCO XGBOOST API TESTS")
        print("=" * 80)
        print(f"API Base URL: {self.base_url}")
        print(f"Test Data Path: {TEST_DATA_PATH}")
        
        test_results = []
        
        # Test 1: Root endpoint
        test_results.append(("Root Endpoint", self.test_root_endpoint()))
        
        # Test 2: Health check
        test_results.append(("Health Check", self.test_health_check()))
        
        # Test 3: Training status
        test_results.append(("Training Status", self.test_training_status()))
        
        # Test 4: List models
        test_results.append(("List Models", self.test_list_models()))
        
        # Test 5: Training (if data file exists)
        if os.path.exists(TEST_DATA_PATH):
            training_id = self.test_training()
            if training_id:
                test_results.append(("Start Training", True))
                
                # Wait for training completion
                training_completed = self.wait_for_training_completion()
                test_results.append(("Training Completion", training_completed))
                
                if training_completed:
                    # Test 6: Single prediction
                    test_results.append(("Single Prediction", self.test_single_prediction()))
                    
                    # Test 7: Batch prediction
                    test_results.append(("Batch Prediction", self.test_batch_prediction()))
                    
                    # Test 8: Validation
                    test_results.append(("Model Validation", self.test_validation()))
                    
                    # Test 9: Feature importance
                    test_results.append(("Feature Importance", self.test_feature_importance()))
                    
                    # Test 10: Performance report
                    test_results.append(("Performance Report", self.test_performance_report()))
                else:
                    print("WARNING: Skipping prediction tests due to training failure")
            else:
                test_results.append(("Start Training", False))
                print("WARNING: Skipping training-dependent tests")
        else:
            print(f"WARNING: Test data file not found: {TEST_DATA_PATH}")
            print("WARNING: Skipping training and prediction tests")
        
        # Print summary
        print("\n" + "=" * 80)
        print("RESULTS: TEST RESULTS SUMMARY")
        print("=" * 80)
        
        passed = 0
        failed = 0
        
        for test_name, result in test_results:
            status = "PASS: PASS" if result else "FAIL: FAIL"
            print(f"{test_name:25} {status}")
            if result:
                passed += 1
            else:
                failed += 1
        
        print("-" * 80)
        print(f"Total Tests: {len(test_results)}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Success Rate: {passed/len(test_results)*100:.1f}%")
        
        overall_success = failed == 0
        if overall_success:
            print("\nSUCCESS: ALL TESTS PASSED - FIRCO XGBOOST API IS WORKING CORRECTLY!")
        else:
            print(f"\nERROR: {failed} TESTS FAILED - PLEASE CHECK THE ERRORS ABOVE")
        
        print("=" * 80)
        return overall_success

def main():
    """Main function to run API tests."""
    print("Firco XGBoost API Tester")
    print("========================")
    
    # Check if API is running
    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=5)
        if response.status_code != 200:
            print(f"FAIL: API not responding at {API_BASE_URL}")
            print("Please make sure the API is running with: python xgb_app_F.py")
            return False
    except requests.exceptions.RequestException:
        print(f"FAIL: Cannot connect to API at {API_BASE_URL}")
        print("Please make sure the API is running with: python xgb_app_F.py")
        return False
    
    # Run tests
    tester = FircoXGBoostAPITester()
    success = tester.run_comprehensive_test()
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 