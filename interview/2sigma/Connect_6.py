"""
This Java code defines a class called "ConnectSix" that represents a game board for the Connect 6 game.
The board is a HashMap with the key being the row number and the value being a HashSet of integers
representing the occupied column numbers. Players can place stones on the board by specifying the row and column
numbers. The game checks for winning conditions by checking for six continuous stones in rows, columns, diagonals,
 and anti-diagonals. If a winning condition is met, the game ends and the winning player is announced.
 The board can be reset to start a new game.
"""

class ConnectSix:
    def __init__(self, m, n):
        self.currentPlayer = 1
        self.m = m
        self.n = n
        self.stones = {}
        print("Game begins. Next Player is 1")

    def resetBoard(self):
        self.currentPlayer = 1
        self.stones = {}
        print("Board has been reset. Next Player is 1")

    def nextPlayer(self):
        self.currentPlayer = 0 if self.currentPlayer == 1 else 1
        print(f"Next player is {self.currentPlayer}")

    def placeStone(self, x, y):
        if self.currentPlayer == 0:
            x, y = -x, -y
        # check if the block has been occupied
        if (x in self.stones and y in self.stones[x]) or (-x in self.stones and -y in self.stones[-x]):
            print("The block has been placed chese, please try again")
            return False
        # check if the row has been occupied, create new row
        if x not in self.stones:
            self.stones[x] = set()
        self.stones[x].add(y)
        print(f"Player {self.currentPlayer} has placed at {abs(x)} {abs(y)}")
        if self.displayWinning(x, y) != -1:
            self.endGame()
        else:
            self.nextPlayer()
        return True

    def endGame(self):
        print("End of Game.")
        self.resetBoard()

    def displayWinning(self, x, y):
        # default no one wins
        winner = -1
        # check 4 directions
        xMove = [1, 0, 1, 1]
        yMove = [0, 1, 1, -1]
        for i in range(4):
            xStart, yStart = x - 6 * xMove[i], y - 6 * yMove[i]
            count = 0
            for j in range(11):
                xStart += xMove[i]
                yStart += yMove[i]
                if xStart in self.stones and yStart in self.stones[xStart]:
                    count += 1
                else:
                    count = 0
                if count == 6:
                    print(f"Winner is {self.currentPlayer}")
                    return self.currentPlayer
        return winner

# Example usage
game = ConnectSix(8, 8)
game.placeStone(1, 0)
game.placeStone(1, 1)
game.placeStone(2, 0)
game.placeStone(2, 2)
game.placeStone(3, 0)
game.placeStone(4, 4)
game.placeStone(4, 0)
game.placeStone(5, 5)
game.placeStone(5, 0)
game.placeStone(6, 5)
game.placeStone(6, 0)
game.placeStone(3, 3)
