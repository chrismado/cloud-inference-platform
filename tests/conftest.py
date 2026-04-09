# Exclude locust_load_test.py from pytest collection.
# Locust imports gevent which monkey-patches SSL, causing RecursionError
# in Python 3.12+ when collected alongside normal tests.
collect_ignore = ["locust_load_test.py"]
