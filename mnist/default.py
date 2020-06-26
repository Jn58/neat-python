import numpy as np
import neat
import multiprocessing
from random import sample
import sys
data = np.fromfile('train_data',dtype=np.uint8,count=28*28*60000,offset=16).reshape(-1,28*28)/255
label = np.fromfile('train_label', dtype=np.uint8, count=60000, offset=8)
onehot = np.eye(10)[label]

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open("default.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  
        self.log.flush()

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass    

sys.stdout = Logger()

def softmax(x):
    x = np.array(x)
    e_x = np.exp(x-x.max())
    return e_x/e_x.sum()

def eval_genome(genome, config):
    index = sample(range(60000),600)
    fitness = float(len(index))
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    for xi, xo in zip(data[index], onehot[index]):
        output = net.activate(xi)

        fitness -= np.square(softmax(output) - xo).sum()
    return fitness / float(len(index))

def run():
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        'config-feedforward')

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)
    # Run until a solution is found.
    winner = p.run(pe.evaluate, 100)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
#    for xi, xo in zip(xor_inputs, xor_outputs):
#        output = winner_net.activate(xi)
#        print("  input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))

if __name__ == '__main__':
    run()