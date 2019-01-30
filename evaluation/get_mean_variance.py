import numpy as np

# values = np.array([5,4.9,4.8,5.3,5.4,5])
# mean = np.mean(values)
# print(mean)
#
# variance = np.var(values)
# print(variance)
#
# variance = 0
# for v in values:
#     variance += (v - mean)**2
# variance /= len(values)
# print(variance)


dataset = '../../../datasets/EuropeanaDates/'
model = 'model.pth'
test_iterations = 10
results_file = open(dataset + '/results/' + model + '/test_dropout_' + str(test_iterations) + '.txt', "r")
output_file = open(dataset + '/results/' + model + '/averaged_prediction.txt', "w")

predicitons = {}
predicitons['predictions'] = np.zeros(test_iterations)

# Read predictions for all iterations
for line in results_file:
    d = line.split(',')
    id = d[0]

    if id not in predicitons:
        predicitons[d[0]]['predictions'] = []
        predicitons[d[0]]['target'] = d[1]

    predicitons[d[0]]['predictions'].append(float(d[2]))

# Compute mean and variance
for k,v in predicitons.iteritems():
    mean = np.mean(v['predictions'])
    var = np.var(v['predictions'])
    output_file.write(k + ',' + v['target'] + ',' + str(mean) + ',' + str(var) + '\n')

