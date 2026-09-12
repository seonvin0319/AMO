"""Small dict-only parameter trees; stable names also define checkpoint keys."""


def map_tree(fn, tree, *others):
    if isinstance(tree, dict):
        return {k: map_tree(fn, v, *(o[k] for o in others)) for k, v in tree.items()}
    return fn(tree, *others)


def flatten(tree, prefix=""):
    out = {}
    for k, value in tree.items():
        key = f"{prefix}/{k}" if prefix else k
        if isinstance(value, dict):
            out.update(flatten(value, key))
        else:
            out[key] = value
    return out


def unflatten(items):
    root = {}
    for path, value in items.items():
        keys = path.split("/")
        target = root
        for key in keys[:-1]:
            target = target.setdefault(key, {})
        target[keys[-1]] = value
    return root
