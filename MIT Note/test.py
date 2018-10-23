class Location(object):
    def __init__(self, x, y):
        '''x and y are floats'''
        self.x = x
        self.y = y
    def move(self, deltaX, deltaY):
        """deltaX and deltaY are floats"""
        return Location(self.x + deltaX, self.y + deltaY)
    def getX(self):
        return self.x
    def getY(self):
        return self.y
    def distFrom(self, other):
        xDist = self.x - other.getX()
        yDist = self.y - other.getY()
        return (xDist**2 + yDist**2)**0.5
    def __str__(self):
        return '<' + str(self.x) + ', ' + str(self.y) + '>'
class Drunk(object):
    def __init__(self, name = None):
        '''Assumes name is a str'''
        self.name = name
    def __str__(self):
        if self != None:
            return self.name
        return 'Anonymouse'
        