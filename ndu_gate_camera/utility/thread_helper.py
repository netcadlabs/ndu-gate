from threading import Lock

locks_lock = Lock()
locks = {}


def get_lock(value):
    global locks
    with locks_lock:
        if value not in locks:
            locks[value] = Lock()
        return locks[value]
