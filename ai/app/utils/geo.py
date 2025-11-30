import math

def is_within(lat1, lng1, lat2, lng2, radius):
    R = 6371000
    dx = math.radians(lng2 - lng1)
    dy = math.radians(lat2 - lat1)
    a = math.sin(dy/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dx/2)**2
    dist = R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return dist <= radius