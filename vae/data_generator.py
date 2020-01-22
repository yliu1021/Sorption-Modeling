import glob
import numpy as np
import os
import random
import time
import threading

class GridDataGenerator:
    def __init__(self, directory='', shift_range=0, rotate=False, flip=False, validation_split=0.0, test_split=0.0, batch_size=32):
        self.directory = directory
        self.shift_range = shift_range
        self.rotate = rotate
        self.flip = flip
        self.batch_size = batch_size

        print('Loading data...')
        self.grids = []
        self.curves = []
        grid_files = glob.glob(os.path.join(directory, 'grids', 'grid*.csv'))
        curve_files = glob.glob(os.path.join(directory, 'results', 'density*.csv'))
        assert len(grid_files) == len(curve_files)
        for curve_file, grid_file in zip(curve_files, grid_files):
            self.grids.append(np.genfromtxt(grid_file, delimiter=','))
            self.curves.append(np.genfromtxt(curve_file, delimiter=',')[:40].reshape(1, 40))
        print('Finished loading data.')

        self.numGrids = len(grid_files)
        numGrids = len(grid_files)
        numGrids *= ((shift_range+1)**2)
        numGrids *= 4 if rotate else 1
        numGrids *= 3 if flip else 1
        self.augmentedNumGrids = numGrids

        numVal = int(self.augmentedNumGrids * validation_split)
        numTest = int(self.augmentedNumGrids * test_split)

        testAndVal = random.sample(range(0, numGrids), numTest+numVal)
        self.validationSet = set(testAndVal[:numVal])
        self.testSet = set(testAndVal[numVal:])

        # self.grid_batch = []
        # self.curve_batch = []
        # self.finished = False
        # self.threadLock = threading.Lock()
        # self.thread = threading.Thread(target=self.setCacheThread)
        # self.thread.start()

    def flow(self):
        while True:
            grid_batch = []
            curve_batch = []
            for _ in range(self.batch_size):
                grid_num = random.randrange(self.augmentedNumGrids)
                while grid_num in self.validationSet or grid_num in self.testSet:
                    grid_num = random.randrange(self.augmentedNumGrids)
                grid, curve = self.xy_pair_for_num(grid_num)
                grid_batch.append(grid.reshape(20, 20, 1))
                curve_batch.append(curve.reshape(40))
            yield np.array(grid_batch), np.array(curve_batch)

            # self.threadLock.acquire()
            # if len(self.grid_batch) > 0 and len(self.curve_batch) > 0:
            #     grid_batch, curve_batch = self.grid_batch.pop(), self.curve_batch.pop()
            #     self.threadLock.release()
            #     try:
            #         yield grid_batch, curve_batch
            #     except GeneratorExit:
            #         self.finished = True
            # else:
            #     self.threadLock.release()
            #     time.sleep(1)

    def xy_pair_for_num(self, grid_num):
        unaugmented_num = grid_num // (self.augmentedNumGrids//self.numGrids)
        grid = self.grids[unaugmented_num]
        curve = self.curves[unaugmented_num]

        augment_factor = 3 if self.flip else 1
        flip_num = grid_num % augment_factor
        grid_num //= augment_factor
        augment_factor = 4 if self.rotate else 1
        rotate_num = grid_num % augment_factor
        grid_num //= augment_factor
        augment_factor = (self.shift_range+1)**2
        shift_num = grid_num % augment_factor

        grid = np.roll(grid, shift_num//(self.shift_range+1), axis=0)
        grid = np.roll(grid, shift_num%(self.shift_range+1), axis=1)
        grid = np.rot90(grid, rotate_num)
        if flip_num == 1: grid = np.flipud(grid)
        if flip_num == 2: grid = np.fliplr(grid)
        if flip_num == 3: grid = np.flipud(np.fliplr(grid))
        return grid, curve

    # def setCacheThread(self):
    #     while not self.finished:
    #         if len(self.curve_batch) > 3:
    #             time.sleep(0.1)
    #         else:
    #             grid_batch = []
    #             curve_batch = []
    #             for _ in range(self.batch_size):
    #                 grid_num = random.randrange(self.augmentedNumGrids)
    #                 while grid_num in self.validationSet or grid_num in self.testSet:
    #                     grid_num = random.randrange(self.augmentedNumGrids)
    #                 grid, curve = self.xy_pair_for_num(grid_num)
    #                 grid_batch.append(grid.reshape(1, 20, 20, 1))
    #                 curve_batch.append(curve.reshape(1, 40))
    #             self.threadLock.acquire()
    #             self.grid_batch.append(np.array(grid_batch))
    #             self.curve_batch.append(np.array(curve_batch))
    #             self.threadLock.release()

# import sys
# sys.path.append('..')
# from simul_dft import *

# train_datagen = GridDataGenerator(directory='../generative_model_3/step_0', shift_range=19, rotate=True, flip=True, validation_split=0.1, test_split=0.1, batch_size=2048)
# time.sleep(1)
# i = 0
# for x_batch, y_batch in train_datagen.flow():
#     print(x_batch.shape)
#     print(y_batch.shape)
#     i += 1
#     if i >= 5:
#         break