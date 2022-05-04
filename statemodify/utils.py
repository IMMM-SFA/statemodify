

def set_alignment(value: str,
                  n_spaces: int = 0,
                  align: str = "left") -> str:
    """Set left or right alignment.

    :param value:                   Value to evaluate.
    :type value:                    str

    :param n_spaces:                Number of spaces to buffer the value by. If less than the length
                                    of the value, no spaces will be added to the padding.
    :type n_spaces:                 int

    :param align:                   Either 'left' or 'right' alignment for the value.
    :type align:                    str

    :return:                        Value with string padding.

    """

    # set align attribute to lower case
    lowercase_align = align.casefold().strip()

    if lowercase_align == "left":
        return f"{value}{n_spaces * ' '}"

    elif lowercase_align == "right":
        return f"{n_spaces * ' '}{value}"

    else:
        raise AssertionError(f"Choice for alignment '{align}' not supported.  Must be 'left' or 'right'.")
