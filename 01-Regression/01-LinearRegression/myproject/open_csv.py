import os


def ospath(path, n=1):
    for i in range(n):
        path = os.path.split(path)[0]
    path = os.path.join(path, '')
    return path
