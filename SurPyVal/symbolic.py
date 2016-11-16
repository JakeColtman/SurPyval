

def covert_to_symbols(matrix, symbol_prefix):
    """
        Map a matrix into symbol world
        Return a matrix the same shape as the input but with symbols at every element

        symbol_prefix: prefix to use to name symbols
    """
    shape = x_s.shape
    name_template = "{0}_{{0}}_{{1}}".format(identifier)
    x_symbols = []
    for x in range(shape[0]):
        for y in range(shape[1]):
            x_symbols.append(symbols(name_template.format(x, y)))
    symbol_array = np.array(x_symbols).reshape(x_s.shape)
    return symbol_array

def create_value_mapping(symbol_array, value_array):
    """
        Create a dict mapping of symbols to values to use to substitute in equations
        Value array and symbol array need to be the same shape
    """
    flat_length = symbol_array.shape[0] * symbol_array.shape[1]
    zipped = zip(list(symbol_array.reshape(flat_length, )), list(value_array.reshape(flat_length, )))
    return {x[0]: x[1] for x in zipped}

def set_values(symbol_formula, symbol_array, value_array):
    """
        Convert an array of symbols into values

        Inverse of convert_to_values
    """
    
    subby = reshape_for_subs(symbol_array, value_array)
    try:
        return symbol_formula.subs(subby)
    except:
        return symbol_formula[0][0].subs(subby)