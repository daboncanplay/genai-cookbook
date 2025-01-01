# Databricks notebook source
# MAGIC %pip install pytest databricks-vectorsearch
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import pytest
import sys

# Skip writing pyc files on a readonly filesystem.
sys.dont_write_bytecode = True

# Run pytest.
retcode = pytest.main([".", "-v", "-p", "no:cacheprovider", "--disable-warnings"])

# Fail the cell execution if there are any test failures.
assert retcode == 0, "The pytest invocation failed. See the log for details."

