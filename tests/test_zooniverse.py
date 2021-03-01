from ChildProject.pipelines.zooniverse import pad_interval

def test_padding():
    assert(pad_interval(300, 800, 500, 1) == (300, 800))
    assert(pad_interval(200, 800, 500, 2) == (0, 1000))

    assert(pad_interval(300, 900, 500, 1) == (100, 1100))

    assert(pad_interval(2000, 2500, 100, 10) == (1750, 2750))

    assert(pad_interval(100, 300, 500, 1) == (-50, 450))
