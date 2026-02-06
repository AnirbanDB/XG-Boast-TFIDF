# Models package for Firco compliance prediction
# Provides organized access to all model implementations

from .tfidf_xgb_F import FircoHierarchicalXGBoost

class ModelFactory:
    """Simple factory for model creation to maintain backward compatibility."""
    
    @staticmethod
    def create_model(model_type: str, **kwargs):
        """
        Create a model instance based on type.
        
        Args:
            model_type: Type of model to create
            **kwargs: Model initialization parameters
            
        Returns:
            Model instance
        """
        model_type_lower = model_type.lower()
        
        if model_type_lower in ["hierarchical_xgboost", "firco_hierarchical_xgb", "tfidf_xgb"]:
            return FircoHierarchicalXGBoost(
                label_encoders=kwargs.get('label_encoders')
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}. Supported: hierarchical_xgboost, firco_hierarchical_xgb, tfidf_xgb")
    
    @staticmethod
    def get_available_models():
        """Get list of available model types."""
        return ["hierarchical_xgboost", "firco_hierarchical_xgb", "tfidf_xgb"]

__all__ = ['FircoHierarchicalXGBoost', 'ModelFactory']
