import torch
import customDatasetTest
import os
import mymodel
import sys


dataset = '../../../datasets/EuropeanaDates/' # Path to dataset
split = 'europeanaDates_test.txt'

test_iterations = 10 # for confidence computing
print("Use Dropout at inference time to get confindence!")

batch_size = 128
workers = 6

model_name = 'model'.strip('.pth')

gpus = [0]
gpu = 0
CUDA_VISIBLE_DEVICES = 0


output_file_path = dataset + 'results/' + model_name + '/test_dropout_' + str(test_iterations) + '.txt'
output_file = open(output_file_path, "w")

state_dict = torch.load(dataset + '/models/' + model_name + '.pth.tar', map_location={'cuda:1':'cuda:0', 'cuda:2':'cuda:0', 'cuda:3':'cuda:0'})


model = mymodel.MyModel()
model = torch.nn.DataParallel(model, device_ids=gpus).cuda(gpu)
model.load_state_dict(state_dict)


test_dataset = customDatasetTest.customDatasetTest(dataset, split, Rescale=299)

for test_it in range(test_iterations):
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True, sampler=None)
    with torch.no_grad():
        model.eval()
        for i, (image, object_features, target, id) in enumerate(test_loader):

            image_var = torch.autograd.Variable(image)
            object_features_var = torch.autograd.Variable(object_features)

            outputs = model(image_var, object_features_var)

            for idx,el in enumerate(outputs):
                regression_output = el
                output_file.write(str(id) + ',' + str(int(target)) + ',' + el + '\n')

    print("Test iteration " + str(test_it) + "completed.")

output_file.close()
