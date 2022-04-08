from scico._version import variable_assign_value

test_var = 12345


def test_var_val():
    var_val = variable_assign_value(__file__, "test_var")
    assert var_val == test_var
