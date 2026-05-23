import pyspacemouse
d = pyspacemouse.open()
s = d.read()
print("state type:", type(s))
print("state fields:", [f for f in dir(s) if not f.startswith("_")])
print("state repr:", s)
d.close()
