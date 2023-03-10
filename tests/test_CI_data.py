import os
def test_CI_Secret():
    CLIENT_ID = os.environ['CLIENT_ID']
    assert len(CLIENT_ID) == 72