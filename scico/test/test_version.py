from scico._version import variable_assign_value

test_var_num = 12345
test_var_str = "12345"


def test_var_val():
    assert variable_assign_value(__file__, "test_var_num") == test_var_num
    assert variable_assign_value(__file__, "test_var_str") == test_var_str
