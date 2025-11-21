
def compare_dictionaries(d1, d2) -> bool:
    if len(d1) != len(d2):
        return False
    for key in d1:
        if key not in d2 or d1[key] != d2[key]:
            return False
    return True