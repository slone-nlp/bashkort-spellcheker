import difflib


def get_aligned_diff(text1, text2, verbose=False, keep_equiv=True):
    """ Get a list of (old,new) segments """
    edits = []
    last_edit = ' '  # space, - or +
    for item in difflib.ndiff(text1, text2):
        diff_type, char = item[0], item[2]
        if verbose:
            print(item)
        if diff_type != ' ':
            if last_edit == ' ':
                edits.append(['', ''])
            if diff_type == '-':
                edits[-1][0] = edits[-1][0] + char
            elif diff_type == '+':
                edits[-1][1] = edits[-1][1] + char
        else:
            if last_edit != ' ' or len(edits) == 0:
                edits.append(['', ''])
            edits[-1][0] = edits[-1][0] + char
            edits[-1][1] = edits[-1][1] + char
        last_edit = diff_type
    return [tuple(t) for t in edits if keep_equiv or t[0] != t[1]]
