from fastapi import FastAPI
from fastapi.param_functions import Query, Path
from vespa.application import Vespa
import models


app = FastAPI()
vespa_app = Vespa(url="http://localhost:8080")
# app_package = ApplicationPackage()


@app.get(
    "/datastore",
    summary="all available datastores",
)
def all_datastores():
    pass


@app.get(
    "/datastore/{datastore_name}/indexs/{index_name}/search",
    summary="Search the documentstore with given query and return top-k documents",
    description="Searches the given datastore with the search strategy specified by the given index \
            and if necessery encodes the query with the specified encoder",
    response_description="The top-K documents",
)
def search(
    datastore_name: str = Path(..., description="Name of the datastore."),
    index_name: str = Path(..., description="Index name."),
    query: str = Query(..., description="The query string."),
    top_k: int = Query(40, description="Number of documents to retrieve."),
    query_encoder: str = Query("dpr", description="Identifier of the query encoder."),
):
    query_embedding = models.encode_query(query_encoder, query)
    body = {
        "query": query,
        "type": "any",
        "datastore_name": datastore_name,
        "queryProfile": index_name,
        "ranking.features.query(query_embedding)": query_embedding,
        "hits": top_k,
    }
    result = vespa_app.query(
        body=body
    )
    return result.json
