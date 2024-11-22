# packet.py
class Packet:
    def __init__(self, id, source, destination, creation_time):
        self.id = id
        self.source = source
        self.destination = destination
        self.creation_time = creation_time
