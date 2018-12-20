import sys

class conf(object):
    def __init__(self, policy, type):
        if policy not in ['cnn', 'lstm']:
            print("incorrect policy {}".format(policy), file=sys.stderr)
            exit(1)
        self.nsteps=2048,
        print(self.nsteps)
        self.nminibatches=1,
        self.lam=0.95,
        self.gamma=0.99,
        self.noptepochs=4,
        self.log_interval=1,
        self.ent_coef=0.001,
        self.lr=lambda f: f*4e-4,
        self.cliprange=lambda f: f*0.2,
        self.total_timesteps=int(3e6),
        self.save_interval=10,
        self.save_dir='model/' + policy,
        self.task_index=0,
        if type == "local":
            self.scope='local_model'
            self.trainable=False

print(conf('cnn', 'model').nsteps)