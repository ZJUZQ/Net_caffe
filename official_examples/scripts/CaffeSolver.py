class CaffeSolver:

    """
    Caffesolver is a class for creating a solver.prototxt file. It sets default
    values and can export a solver parameter file.
    Note that all parameters are stored as strings. Strings variables are
    stored as strings in strings.
    """

    def __init__(self, testnet_prototxt_path="testnet.prototxt",
                 trainnet_prototxt_path="trainnet.prototxt", debug=False):

        self.sp = {}

        # critical:
        self.sp['base_lr'] = '0.001'
        self.sp['momentum'] = '0.9' # Set momentum to accelerate learning by taking weighted average of current and previous updates.

        # speed:
        self.sp['test_iter'] = '100'
        self.sp['test_interval'] = '250'

        # looks:
        self.sp['display'] = '25' # Display the current training loss and accuracy every 25 iterations.
        self.sp['snapshot'] = '2500' # Snapshots are files used to store networks we've trained. We'll snapshot every 2500 iterations
        self.sp['snapshot_prefix'] = '"snapshot"' # string withing a string!

        # learning rate policy
        self.sp['lr_policy'] = '"fixed"' # Set `lr_policy` to define how the learning rate changes during training.

        # important, but rare:
        self.sp['gamma'] = '0.1'
        #self.sp['power'] = '0.75'
        self.sp['weight_decay'] = '0.0005'  # Set weight decay to regularize and prevent overfitting
        self.sp['train_net'] = '"' + trainnet_prototxt_path + '"'
        self.sp['test_net'] = '"' + testnet_prototxt_path + '"'

        # pretty much never change these.
        self.sp['max_iter'] = '100000'
        self.sp['test_initialization'] = 'false'
        self.sp['average_loss'] = '25'  # this has to do with the display.
        self.sp['iter_size'] = '1'  # this is for accumulating gradients

        if (debug):
            self.sp['max_iter'] = '12'
            self.sp['test_iter'] = '1'
            self.sp['test_interval'] = '4'
            self.sp['display'] = '1'

    def add_from_file(self, filepath):
        """
        Reads a caffe solver prototxt file and updates the Caffesolver
        instance parameters.
        """
        with open(filepath, 'r') as f:
            for line in f:
                if line[0] == '#':
                    continue
                splitLine = line.split(':')
                self.sp[splitLine[0].strip()] = splitLine[1].strip()

    def write(self, filepath):
        """
        Export solver parameters to INPUT "filepath". Sorted alphabetically.
        """
        f = open(filepath, 'w')
        for key, value in sorted(self.sp.items()):
            if not(type(value) is str):
                raise TypeError('All solver parameters must be strings')
            f.write('%s: %s\n' % (key, value))