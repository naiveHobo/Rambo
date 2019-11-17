class Configuration:

    def __init__(self, args):
        self.mode = args["mode"]
        self.resume = args["resume"]
        self.train_dir = args['train_dir']
        self.data_dir = args['data_dir']
        self.log_dir = args['log_dir']
        self.epochs = args['epochs']
        self.batch_size = args['batch_size']
        self.learning_rate = args['learning_rate']
        self.dropout = args['dropout']
        self.beta_l2 = args['beta_l2']
        self.visualize = args['visualize']
        self.save_model = args['save_model']
        self.model = args['model']
