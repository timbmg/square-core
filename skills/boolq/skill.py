import logging

from square_skill_api.models import QueryOutput, QueryRequest

from square_skill_helpers import ModelAPI

logger = logging.getLogger(__name__)

model_api = ModelAPI()


async def predict(request: QueryRequest) -> QueryOutput:
    """Predicts yes/no for a boolean question with context"""
    query = request.query
    context = request.skill_args["context"]
    prepared_input = [context, query]

    model_request = {
        "input": prepared_input,
        "preprocessing_kwargs": {},
        "model_kwargs": {},
        "adapter_name": "AdapterHub/bert-base-uncased-pf-boolq",
    }
    model_api_output = await model_api(
        model_name="bert-base-uncased",
        pipeline="sequence-classification",
        model_request=model_request,
    )
    logger.info(f"Model API output:\n{model_api_output}")

    return QueryOutput.from_sequence_classification(
        answers=["No", "Yes"], model_api_output=model_api_output, context=context
    )
