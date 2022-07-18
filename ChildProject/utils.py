import os
from datetime import datetime

def path_is_parent(parent_path: str, child_path: str):
    # Smooth out relative path names, note: if you are concerned about symbolic links, you should use os.path.realpath too
    parent_path = os.path.abspath(parent_path)
    child_path = os.path.abspath(child_path)

    # Compare the common path of the parent and child path with the common path of just the parent path. Using the commonpath method on just the parent path will regularise the path name in the same way as the comparison that deals with both paths, removing any trailing path separator
    return os.path.commonpath([parent_path]) == os.path.commonpath(
        [parent_path, child_path]
    )


class Segment:
    def __init__(self, start, stop):
        self.start = start
        self.stop = stop

    def length(self):
        return self.stop - self.start

    def __repr__(self):
        return "Segment([{}, {}])".format(self.start, self.stop)


def intersect_ranges(xs, ys):
    # Try to get the first range in each iterator:
    try:
        x, y = next(xs), next(ys)
    except StopIteration:
        return

    while True:
        # Yield the intersection of the two ranges, if it's not empty:
        intersection = Segment(max(x.start, y.start), min(x.stop, y.stop))
        if intersection.length() > 0:
            yield intersection

        # Try to increment the range with the earlier stopping value:
        try:
            if x.stop <= y.stop:
                x = next(xs)
            else:
                y = next(ys)
        except StopIteration:
            return

class TimeInterval:
    def __init__(self, start : datetime, stop : datetime):
        #remove the day/month/year component
        self.start = start.replace(year=1900, month=1, day=1)
        self.stop = stop.replace(year=1900, month=1, day=1)

    def length(self):
        return self.stop - self.start

    def __repr__(self):
        return "TimeInterval([{}, {}])".format(self.start, self.stop)
    
def time_intervals_intersect(ti1 : TimeInterval, ti2 : TimeInterval):
    """
    given 2 time intervals (those do not take in consideration days, only time in the day), return an array of new interval(s) representing the intersections of the original ones.
    eg :
        - time_intervals_intersect(TimeInterval(datetime(1900,1,1,8,57),datetime(1900,1,1,21,4)),TimeInterval(datetime(1900,1,1,10,36),datetime(1900,1,1,22,1))) => [TimeInterval(10:36 , 21:04)]
        - time_intervals_intersect(TimeInterval(datetime(1900,1,1,8,57),datetime(1900,1,1,22,1)),TimeInterval(datetime(1900,1,1,21,4),datetime(1900,1,1,10,36))) => [TimeInterval(08:57 , 10:36),TimeInterval(21:04 , 22:01)]
    """
    #The calculation and boolean evaluation is done that way to optimize the process, those expressions were obtained using a Karnaugh table. Given the relations between the different start and ending times, the boolean relations used below gives the correct intervals
    a = ti1.start <= ti1.stop
    b = ti2.start <= ti2.stop
    c = ti1.stop <= ti2.stop
    d = ti1.start <= ti2.start
    e = ti1.start <= ti2.stop
    f = ti2.start <= ti1.stop
    r = []
    #case where correct resulting interval is [start of the 2nd interval : end of the 1st interval]
    if c and (d and (not e or f) or not e and f) or d and not e and f : r = [TimeInterval(ti2.start,ti1.stop)]
    #case where correct resulting interval is [start of the 2nd interval : end of the 2nd interval]
    elif not c and (d and (b or not a) or not a and b) or not a and b and d : r = [ti2]
    #case where correct resulting interval is [start of the 1st interval : end of the 2nd interval]
    elif not c and (not d and (not e and not f or e) or e and not f) or not d and e and not f : r = [TimeInterval(ti1.start,ti2.stop)]
    #case where correct resulting interval is [start of the 1st interval : end of the 1st interval]
    elif c and (not d and (not a and not b or a) or a and not b) or a and not b and not d : r = [ti1]
    # !! here the expression was simplified because previous statements should already have caught their relevant cases (ie this statement should always be last or changed)
    #case where correct resulting interval is [start of the 1st interval : end of the 2nd interval] U [start of the 2nd interval : end of the 1st interval]
    elif not a and (not b or e) or d and e and f : r = [TimeInterval(ti1.start,ti2.stop),TimeInterval(ti2.start,ti1.stop)]
    
    #remove the intervals having equal values (3:00 to 3:00)
    i = 0
    while i < len(r):
        if r[i].start == r[i].stop : r.pop(i)
        else : i += 1
    return r

def get_audio_duration(filename):
    import sox

    if not os.path.exists(filename):
        return 0

    duration = 0
    try:
        duration = sox.file_info.duration(filename)
    except:
        pass

    return duration
