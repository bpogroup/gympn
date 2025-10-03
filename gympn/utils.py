# this file contains the utility functions used in the project

def sim_tokens_values_from_bindings(bindings):
    """
    Convert a list of bindings to a list of (string-like) token values.

    Parameters
    ----------
    tokens : list
        List of token bindings.

    Returns
    -------
    dict
        Dict having transition names as keys and the corresponding enabling bindings as values.
    """
    #return [[ei[1].value for ei in ee] for ee in [ee[0] for ee in bindings]]
    ret = {}
    keys = set([el[2]._id for el in bindings])
    for k in keys:
        ret[k] = [el[0] for el in bindings if el[2]._id == k]

    return ret

def binding_from_tokens_values(tokens_values, bindings):
    """
    Convert a 1-element list of (string-like) token values (a single binding) to a list of bindings by extracting them from a pre-existing list.

    Parameters
    ----------
    tokens_values : list
        List of (string-like) token values.
    bindings : list
        List of bindings.

    Returns
    -------
    list
        List of bindings.
    """

    return [bindings[i] for i in range(len(bindings)) if (bindings[i][2], bindings[i][0]) == tokens_values][0]
