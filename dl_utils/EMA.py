import torch
import os
import warnings


class EMA:
    # Found this useful (thanks alexis-jacq):
    # https://discuss.pytorch.org/t/how-to-apply-exponential-moving-average-decay-for-variables/10856/3
    def __init__(self, gamma=0.99, save=True, save_frequency=5, save_filename="ema_weights.pth"):
        """
        Initialize the weight to which we will do the
        exponential moving average and the dictionary
        where we store the model parameters
        """
        self.gamma = gamma
        self.registered = {}
        self.save_filename = save_filename
        self.save_frequency = save_frequency
        self.count = 0

        if not save:
            warnings.warn("Note that the exponential moving average weights will not be saved to a .pth file!")

        if save_filename in os.listdir("."):
            self.registered = torch.load(self.save_filename)

    def __call__(self, model):
        self.count += 1
        for name, param in model.named_parameters():
            if param.requires_grad:
                new_weight = param.clone().detach() if name not in self.registered else self.gamma * param + (1 - self.gamma) * self.registered[name]
                self.registered[name] = new_weight

        if self.count % self.save_frequency == 0:
            self.save_ema_weights()

    def copy_weights_to(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data = self.registered[name]

    def save_ema_weights(self):
        torch.save(self.registered, self.save_filename)



#start = torch.cuda.Event(enable_timing=True)
#end = torch.cuda.Event(enable_timing=True)
#
#start.record()
#for _ in range(100):
#    vgg(x)
#    #for name, param in vgg.named_parameters():
#    #    pass
#end.record()
#
## Waits for everything to finish running
#torch.cuda.synchronize()
#print(start.elapsed_time(end))
#
#for _ in range(10):
#    ema(vgg)
#    #vgg(x)

#model = nn.Linear(1000, 1)
#ema = EMA()
#ema(model)  # this registers the weights
#x = torch.randn((1, 1000)) * 10000
#y = torch.tensor([1]).float().unsqueeze(0)
#loss_fn = nn.MSELoss()
#optimizer = optim.Adam(model.parameters(), lr=0.1)
#
## Train Network
#for _ in range(10):
#    start = time.time()
#    for epoch in range(10000):
#        # forward
#        scores = model(x)
#        loss = loss_fn(scores, y)
#
#        # backward
#        optimizer.zero_grad()
#        loss.backward()
#
#        # gradient descent or adam step
#        optimizer.step()
#
#        #ema(model)  # this updates the registered weights with exponential moving average
#
#    print(f"Time took {time.time() - start}")
#
#
#
#
#