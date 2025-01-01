import pytest
import pyspark
from pyspark.sql import SparkSession
from databricks.vector_search.client import VectorSearchClient
import mlflow

# Because this file is not a Databricks notebook, you
# must create a Spark session. Databricks notebooks
# create a Spark session for you by default.
spark = SparkSession.builder \
                    .appName('integrity-tests') \
                    .getOrCreate()

# Get the vector search index
vsc = VectorSearchClient(disable_notice=True)
index = vsc.get_index(endpoint_name="daboncanplay_vector_search", index_name="ps_ci_cd.default.my_agent_app_poc_chunked_docs_gold_index")

challenger_chain_name = "models:/ps_ci_cd.default.my_agent_app@Challenger"


# Return results from vs index
def vs_returns_results():
    results = index.similarity_search(columns=["chunked_text", "chunk_id", "path"], query_text="Can I turn right on a red light?")
    return results["result"]["row_count"]

# Test that vs index returned 5 results
def test_returns_results():
    assert vs_returns_results() == 5

def vs_returns_correct_results():
    results = index.similarity_search(columns=["chunked_text", "chunk_id", "path"], query_text="Do I need a license to sell insurance?")
    return results["result"]["data_array"][0][2]

# Test that vs index returned 5 results
def test_returns_correct_results():
    assert vs_returns_correct_results() == "dbfs:/Volumes/mind_constructor/default/utah_code/Title 32B.pdf"

# Test that vs index returned 5 results
def test_returns_challenger_chain():
    assert mlflow.langchain.load_model(challenger_chain_name)

def vs_challenger_returns_correct_results():
    challenger_chain = mlflow.langchain.load_model(challenger_chain_name)
    chain_input = {
        "messages": [
            {
                "role": "user",
                "content": "Can I turn right on a red light?", 
            }
        ]
    }
    results = challenger_chain.invoke(chain_input)
    
    return results.find("Yes")

def test_challenger_returns_correct_results():
    assert vs_challenger_returns_correct_results() >= 0





