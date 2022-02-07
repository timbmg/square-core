import logging
import uuid

from square_skill_api.models.prediction import QueryOutput
from square_skill_api.models.request import QueryRequest

from square_skill_helpers.config import SquareSkillHelpersConfig
from square_skill_helpers.square_api import ModelAPI

logger = logging.getLogger(__name__)

config = SquareSkillHelpersConfig.from_dotenv()
model_api = ModelAPI(config)

async def predict(request: QueryRequest) -> QueryOutput:
    query = request.query
    context = request.skill_args["context"]
    prepared_input = [context, query] 
    
    model_request = { 
        "input": prepared_input,
        "preprocessing_kwargs": {},
        "model_kwargs": {},
        "adapter_name": "AdapterHub/narrativeqa"
    }
    model_api_output = await model_api(
        model_name=request.skill_args["base_model"],
        pipeline="TextGenerationPipeline", 
        model_request=model_request
    )
    logger.info(f"Model API output:\n{model_api_output}")

    return QueryOutput.PredictionOutputForGeneration(
        model_api_output=model_api_output, 
        context=context
    )
