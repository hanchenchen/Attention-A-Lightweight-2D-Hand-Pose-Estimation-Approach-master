import argparse
import json
import os
from model import create_model
from load_tfrecord import load_dataset, load_training_dataset, load_xyz_dataset
import tensorflow as tf
from model import create_model
import time
from pck import get_pck_with_sigma
import numpy as np
start = time.time()
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
parser = argparse.ArgumentParser(description='use the specified dataset to train the model.')
parser.add_argument('dataset_name', type=str, default='FreiHAND_pub_v2',
                    help='choose one dataset.')
args = parser.parse_args()
choosed = ['FreiHAND_pub_v2', 'Panoptic']
configs = json.load(open('configs/' + args.dataset_name + '.json'))
filepath = args.dataset_name + '/weights.hdf5'
model = create_model()
model.load_weights(filepath)
predictions = {}
ground_truth = {}
names, images, labels = load_xyz_dataset(args.dataset_name, 'testing')
results = model.predict(images)
names = [''.join(str(j) for j in i) for i in list(names.as_numpy_iterator())]
results = (results*224).tolist()
labels = [(i[0]*224).tolist() for i in list(labels.as_numpy_iterator())]
print('results:', len(results))
# print(names[0],results[0],labels[0])
for i in range(len(results)):
    predictions[names[i]] = {'prd_label': results[i], 'resol': 224}
    ground_truth[names[i]] = labels[i]
json.dump(predictions, open(args.dataset_name + '/predictions.json', 'w'))
json.dump(ground_truth, open(args.dataset_name + '/ground_truth.json', 'w'))
pck_results = get_pck_with_sigma(predictions, ground_truth)
print(pck_results)
json.dump(pck_results, open(args.dataset_name + '/pck_results.json', 'w'))
'''print(results)
for i in range(names):
    print(tf.convert_to_tensor(images))
    predictions[names[i].tostring()] = {'prd_label': list(model.predict(np.array(images))), 'resol':224}
    ground_truth[names[i].tostring()] = list(labels[i])
    print(predictions[names[i].tostring()], ground_truth[names[i].tostring()])
    break'''
end = time.time()
print('predicted done in',end - start, 'sec.')
'''
# Evaluate the model on the test data using `evaluate`
print("Evaluate on test data...")
evaluate_fpath = open(os.path.join(args.dataset_name, 'evaluate.json'), 'w')
results = json.load(evaluate_fpath)
results[configs['learning_rate']] = model.evaluate(testing_images, ground_truth)
print("test loss, test acc:", results)
json.dump(results, evaluate_fpath)
evaluate_fpath.close()


# Generate predictions (probabilities -- the output of the last layer)
# on new data using `predict`
print("Generate predictions: ", testing_images.shape)
predictions = model.predict(testing_images)#, steps = 100)
# print("predictions shape:", predictions.shape)
# tf.print(predictions)

# Save the predictions to (args.dataset_name)/predictions.json
predictions_fpath = open(os.path.join(args.dataset_name, 'predictions.json'), 'w')
pred_list = [i.tolist() for i in predictions]
pred_list = {testing[i]:pred_list[i] for i in range(len(testing))}
json.dump(pred_list, predictions_fpath)
predictions_fpath.close()
'''