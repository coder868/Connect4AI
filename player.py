import numpy as np
from gameboard import Gameboard


board = Gameboard()


print(board)
move = 0
while(True): 
    while(board.is_done() == False):
        if move % 2 == 0:
            player = '1'
        else:
            player = '2'
            
        print(board)
        
        print('Turn: Player '+player)
        column = int(input('Column: '))
        column = column - 1
        board.turn(int(player), column)
        move += 1
    print('\nGame Over   Winner: Player ' + str(board.winner))
    board.reset()
    move = 0