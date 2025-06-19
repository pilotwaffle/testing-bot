# api/routers/ml_routes.py
import logging

from fastapi import APIRouter, Form, Depends, HTTPException

from main import get_trading_engine_dep # Import dependency providers from main
from ml.models import MLTrainRequest # Import Pydantic model

router = APIRouter(
    prefix="/api/ml",
    tags=["Machine Learning"]
)

logger = logging.getLogger(__name__)

@router.post("/train")
async def api_ml_train(
    model_type: str = Form(...), # Use Form for direct HTML form submission
    symbol: str = Form("BTC/USDT"),
    engine=Depends(get_trading_engine_dep)
):
    """Trains a specified ML model for a given symbol."""
    logger.info(f"Starting ML training request: model='{model_type}', symbol='{symbol}'")

    train_request = MLTrainRequest(model_type=model_type, symbol=symbol)

    # Fetch training data
    logger.info(f"Fetching OHLCV data for {train_request.symbol}")
    df = await engine.data_fetcher.fetch_ohlcv(train_request.symbol, limit=1000) # Await async call
    logger.info(f"Fetched {len(df)} data points for {train_request.symbol}")

    if df.empty:
        logger.warning(f"No data available for {train_request.symbol}. Cannot train.")
        raise HTTPException(status_code=400, detail=f"No sufficient data available for {train_request.symbol} to train the model.")

    # Train the specified model
    logger.info(f"Calling ML engine to train {train_request.model_type} model.")
    if train_request.model_type == "neural_network":
        result = engine.ml_engine.train_neural_network(train_request.symbol, df)
    elif train_request.model_type == "lorentzian":
        result = engine.ml_engine.train_lorentzian_classifier(train_request.symbol, df)
    elif train_request.model_type == "social_sentiment":
        # Social sentiment usually doesn't directly use OHLCV from data_fetcher;
        # this is a simulated method in ml_engine.
        result = engine.ml_engine.train_social_sentiment_analyzer(train_request.symbol)
    elif train_request.model_type == "risk_assessment":
        result = engine.ml_engine.train_risk_assessment_model(train_request.symbol, df)
    else:
        logger.error(f"Invalid model type requested: {train_request.model_type}")
        raise HTTPException(status_code=400, detail="Invalid model type. Choose: neural_network, lorentzian, social_sentiment, risk_assessment.")

    if not result.get("success"):
        logger.error(f"ML training failed for {model_type}: {result.get('error')}")
        raise HTTPException(status_code=500, detail=f"ML training failed: {result.get('error', 'Unknown error')}")

    logger.info(f"ML training completed for {model_type}: {result}")
    return result

@router.get("/test")
async def api_ml_test(engine=Depends(get_trading_engine_dep)):
    """Performs a basic diagnostic test of the ML system dependencies."""
    try:
        # Import directly to avoid circular imports at top level if not needed elsewhere
        import sklearn, numpy, pandas
        
        # Test basic functionality (e.g., model creation, simple fit)
        X = numpy.random.randn(100, 3)
        y = numpy.random.choice([0, 1], 100)
        
        # Using a simple model that doesn't need scaling or complex features
        from sklearn.tree import DecisionTreeClassifier # Simpler than RandomForest for quick test
        model = DecisionTreeClassifier(random_state=42)
        model.fit(X, y)
        accuracy = model.score(X, y)
        
        return {
            "success": True,
            "message": "ML system dependencies are working correctly.",
            "test_accuracy": f"{accuracy:.1%}",
            "scikit_learn_version": sklearn.__version__,
            "numpy_version": numpy.__version__,
            "pandas_version": pandas.__version__
        }
    except ImportError as ie:
        logger.error(f"ML test failed due to missing dependency: {ie}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"ML test failed: Missing dependency - {ie}. Please install all required ML libraries.")
    except Exception as e:
        logger.error(f"ML system test encountered an unexpected error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"ML test failed: {str(e)}")

@router.get("/status")
async def api_ml_status(engine=Depends(get_trading_engine_dep)):
    """Retrieves the status (including training history) of all ML models."""
    status = engine.ml_engine.get_model_status()
    return {"success": True, "models": status}