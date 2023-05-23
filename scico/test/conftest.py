def pytest_addoption(parser):
    parser.addoption(
        "--level", action="store", default=1, type=int, help="Set test level to be run"
    )
