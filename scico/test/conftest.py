def pytest_addoption(parser):
    # Level definitions:
    #  1  Critical tests only
    #  2  Skip tests that do have a significant impact on coverage
    #  3  All tests
    parser.addoption(
        "--level", action="store", default=3, type=int, help="Set test level to be run"
    )
