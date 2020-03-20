class Arena:
    """
    Represents the terrain. Ex: 30x30 ft grid
    """
    def __init__(self, name, *args, **kwargs):
        self.name = name
    pass

class SquareRoom(Arena):
    """
    30 x 30 ft room
    """
    def __init__(self, *args, **kwargs):
       super().__init__(name="Square Room", *args, **kwargs)

    def check_if_legal(self, target_location):
        """
        Checks to see if movement is legal

        Todo: Implement this
        """
        x = target_location[0]
        y = target_location[1]
        if (x < 0) or (x > 30):
            return False
        elif (y < 0) or (y > 30):
            return False
        else:
            return True

# Todo: Place this into DB
square_room = SquareRoom()
