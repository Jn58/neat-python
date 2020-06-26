import numpy as np
import neat
import multiprocessing
data = np.fromfile('train_data',dtype=np.uint8,count=28*28*60000,offset=16).reshape(-1,28*28)/255
label = np.fromfile('train_label', dtype=np.uint8, count=60000, offset=8)
onehot = np.eye(10)[label]

data = data[:100]
onehot = onehot[:100]

def softmax(x):
    x = np.array(x)
    e_x = np.exp(x-x.max())
    return e_x/e_x.sum()

def eval_genome(genome, config):
    fitness = float(data.shape[0])
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    for xi, xo in zip(data, onehot):
        output = net.activate(xi)

        fitness -= np.square(softmax(output) - xo).sum()
    return fitness / float(data.shape[0])

def run():
    # Load configuration.
    config = neat.Config(neat.SharedGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        'config-feedforward')

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(False))

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