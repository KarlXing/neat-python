from neat import nn

# TODO: These tests are just smoke tests to make sure nothing has become badly broken.  Expand
# to include more detailed tests of actual functionality.

def create_simple():
    neurons = [nn.Neuron('INPUT', 1, 0.0, 5.0, 'exp'),
               nn.Neuron('HIDDEN', 2, 0.0, 5.0, 'exp'),
               nn.Neuron('OUTPUT', 3, 0.0, 5.0, 'exp')]
    connections = [(1, 2, 0.5), (1, 3, 0.5), (2, 3, 0.5)]

    return nn.Network(neurons, connections, 1)


def test_manual_network():
    net = create_simple()
    net.serial_activate([0.04])
    net.parallel_activate([0.04])
    repr(net)