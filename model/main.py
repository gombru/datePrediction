import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data.distributed
import customDataset
import trainingFunctions as t
import mymodel

from pylab import zeros, arange, subplots, plt, savefig

training_id = 'datePrediction'
dataset = '../../../datasets/EuropeanaDates/' # Path to dataset
split_train = 'europeanaDates_train.txt'
split_val =  'europeanaDates_val.txt'
ImgSize = 299
gpus = [0]
gpu = 0
workers = 4 # Num of data loading workers
epochs = 301
start_epoch = 0 # Useful on restarts
batch_size = 32 #256 #32 Batch size
print_freq = 1 #25
resume = None
# resume = dataset + 'models/model.pth.tar'
# evaluate = False # Evaluate model on validation set at start
plot = True
best_score = 0
best_epoch = 0
best_loss = 100


optimizer_name = 'ADAM'
if optimizer_name == 'ADAM':
    print("Using ADAM optimizer")
    lr = 1e-6
else:
    print("Using SGD optimizer")
    lr = 1e-3
    decay_every = 30 # Decay lr by a factor of 10 every decay_every epochs
    momentum = 0.9
    weight_decay = 1e-4


model = mymodel.MyModel(gpu=gpu)

# define loss function (criterion) and optimizer
criterion = nn.CrossEntropyLoss().cuda(gpu)
# criterion = nn.BCEWithLogitsLoss().cuda(gpu)

# OPTIMIZER
# ADAM
if optimizer_name == 'ADAM':
    print("Using ADAM optimizer with lr: " + str(lr))
    optimizer = torch.optim.Adam(lr = lr)
# SGD
else:
    print("Using SGD optimizer")
    optimizer = torch.optim.SGD(lr,momentum=momentum,weight_decay=weight_decay)

model = torch.nn.DataParallel(model, device_ids=gpus).cuda(gpu)

# Freeze layers
# for param in model.parameters():
#     param.requires_grad = False # This would froze all net
# Parameters of newly constructed modules have requires_grad=True by default


# optionally resume from a checkpoint
if resume:
    print("Loading pretrained model")
    # if os.path.isfile(resume):
    print("=> loading checkpoint '{}'".format(resume))
    checkpoint = torch.load(resume, map_location={'cuda:1':'cuda:0', 'cuda:2':'cuda:0', 'cuda:3':'cuda:0'})
    #start_epoch = checkpoint['epoch']
    #best_score = checkpoint['best_score']
    model.load_state_dict(checkpoint, strict=False)
    #optimizer.load_state_dict(checkpoint['optimizer'])
    #print("=> loaded checkpoint '{}' (epoch {})"
          # .format(resume, checkpoint['epoch']))
    print("Checkpoint loaded")
    # else:
    #     print("=> no checkpoint found at '{}'".format(resume))

cudnn.benchmark = True

# Data loading code
train_dataset = customDataset.CustomDataset(
    dataset,split_train,Rescale=0,RandomCrop=ImgSize,Mirror=True)

val_dataset = customDataset.CustomDataset(
    dataset, split_val,Rescale=ImgSize,RandomCrop=0,Mirror=False)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True,
    num_workers=workers, pin_memory=True)

val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

plot_data = {}
plot_data['train_loss'] = zeros(epochs)
plot_data['val_loss'] = zeros(epochs)
plot_data['epoch'] = 0

# if evaluate:
#     t.validate(val_loader, model, criterion, print_freq, plot_data)

it_axes = arange(epochs)

_, ax1 = subplots()
ax2 = ax1.twinx()
ax1.set_xlabel('epoch ' + " (GPU " + str(gpu) +") ")
ax1.set_ylabel('train loss (r), val loss (y)')
# ax2.set_ylabel('train acc avg (b), val acc avg (g)')
# ax2.set_autoscaley_on(False)
# ax1.set_ylim([0.3, 0.8])
# ax2.set_ylim([49, 101])


for epoch in range(start_epoch, epochs):
    plot_data['epoch'] = epoch
    if optimizer_name == 'SGD':
        lr = t.adjust_learning_rate(optimizer, epoch, lr, decay_every)

    # train for one epoch
    plot_data = t.train(train_loader, model, criterion, optimizer, epoch, print_freq, plot_data, gpu)

    # evaluate on validation set
    plot_data = t.validate(val_loader, model, criterion, print_freq, plot_data, gpu)

    # remember best prec@1 and save checkpoint
    # is_best = plot_data['val_acc_avg'][epoch] > best_score
    # if is_best and epoch != 0:
    #     print("New best model. Val acc = " + str(plot_data['val_acc_avg'][epoch]))
    #     best_score = max(plot_data['val_acc_avg'][epoch], best_score)
    #     best_epoch = epoch
    #     ax1.set_xlabel('epoch ' + ' / GPU: ' + str(gpu) + ' / Best epoch: ' + str(best_epoch) + ' with Val Acc: ' + str(best_score))
    #     t.save_checkpoint(dataset, model, is_best, filename = dataset +'/models/' + training_id + '_epoch_' + str(epoch) + '_ValAcc_' + str(int(plot_data['val_acc_avg'][epoch])))

    # Save checkpoint by loss
    is_best = plot_data['val_loss'][epoch] < best_loss
    if is_best and epoch != 0:
        print("New best model by loss. Loss = " + str(plot_data['val_loss'][epoch]))
        best_loss = plot_data['val_loss'][epoch]
        # t.save_checkpoint(dataset, model, is_best, filename = dataset +'/models_loss/' + training_id + '_epoch_' + str(epoch) + '_ValAcc_' + str(int(plot_data['val_acc_avg'][epoch])) + '_ValLoss_' + str(round(plot_data['val_loss'][epoch],2)))
        t.save_checkpoint(dataset, model, is_best, filename = dataset +'/models_loss/' + training_id + '_epoch_' + str(epoch) + '_ValLoss_' + str(round(plot_data['val_loss'][epoch],2)))

    if plot:
        ax1.plot(it_axes[0:epoch], plot_data['train_loss'][0:epoch], 'r')
        # ax2.plot(it_axes[0:epoch], plot_data['train_acc_avg'][0:epoch], 'b')
        #
        # ax2.plot(it_axes[0:epoch], plot_data['train_acc_hate'][0:epoch], 'c')
        # ax2.plot(it_axes[0:epoch], plot_data['train_acc_notHate'][0:epoch], '#DBA901')

        ax1.plot(it_axes[0:epoch], plot_data['val_loss'][0:epoch], 'y')
        # ax2.plot(it_axes[0:epoch], plot_data['val_acc_avg'][0:epoch], 'g')
        #
        # ax2.plot(it_axes[0:epoch], plot_data['val_acc_hate'][0:epoch], 'k')
        # ax2.plot(it_axes[0:epoch], plot_data['val_acc_notHate'][0:epoch], 'm')

        plt.title(training_id)
        plt.ion()
        plt.grid(True)
        plt.show()
        plt.pause(0.001)

        if epoch % 10 == 0 and epoch != 0:
            title = dataset +'/models/training/' + training_id + '_epoch_' + str(epoch) + '.png'  # Save graph to disk
            savefig(title, bbox_inches='tight')

