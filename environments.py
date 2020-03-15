class Arena:
    """
    Represents the terrain. Ex: 30x30 ft grid
    """
    pass

class SquareRoom(Arena):
    """
    30 x 30 ft room
    """
    def check_if_legal(self, target_location):
        """
        Checks to see if movement is legal

        Todo: Implement this
        """
        return True

# Todo: Place this into DB
square_room = SquareRoom()
