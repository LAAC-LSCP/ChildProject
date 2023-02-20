import os
import shutil

TO_REMOVE = [os.path.join('examples','valid_raw_data','metadata','annotations.csv'), os.path.join('output')]

def pytest_sessionstart(session):
    """
    Called after the Session object has been created and
    before performing collection and entering the run test loop.
    """
    for path in TO_REMOVE:
        if os.path.exists(path):
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)

