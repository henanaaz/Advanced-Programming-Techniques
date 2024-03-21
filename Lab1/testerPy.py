import numpy as np

def check_sudoku(grid):
    """ Return True if grid is a valid Sudoku square, otherwise False. """
    for i in range(9):
        # j, k index top left hand corner of each 3x3 tile
        j, k = (i // 3) * 3, (i % 3) * 3
        if len(set(grid[i,:])) != 9 or len(set(grid[:,i])) != 9\
                   or len(set(grid[j:j+3, k:k+3].ravel())) != 9:
            return False
    return True

sudoku = """193672485
            462358971
            785914623
            538296714
            674135298
            219487356
            826741539
            941523867
            357869142"""
# Turn the provided string, sudoku, into an integer array
grid = np.array([[int(i) for i in line] for line in sudoku.split()])
print(grid)

if check_sudoku(grid):
    print('grid valid')
else:
    print('grid invalid')