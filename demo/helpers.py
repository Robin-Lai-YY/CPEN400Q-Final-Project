import matplotlib.pyplot as plt

def grid_imshow(title, rows, cols, w, h, grid):
    fig, ax = plt.subplots(rows, cols, figsize=(2*cols,2*rows))
    fig.suptitle(title)
    for row in range(rows):
        for col in range(cols):
            ax[row,col].matshow(grid[row,col].reshape((w,h)), cmap='Purples')
    return fig
